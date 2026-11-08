"""Enrich saved compute profiles with independently measured collective overlap."""

from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from typing import Any

import torch
import torch.distributed as dist

from shadowspill.artifacts._values import digest, plain
from shadowspill.artifacts.device import require_compiled_device
from shadowspill.artifacts.profiled_program import ProfiledProgram
from shadowspill.artifacts.profiles import Profiles, TaskProfile
from shadowspill.artifacts.tasks import stage_jobs
from shadowspill.communication.binding.backend import shared_backend_setup
from shadowspill.communication.binding.groups import GroupBindings
from shadowspill.communication.info import CommunicationInfo
from shadowspill.communication.overlap import OverlapRoute, overlap_routes
from shadowspill.communication.requests import collective_requests
from shadowspill.ir import TaskOverlapCost
from shadowspill.preparation.lifecycle import LoadedPreparations
from shadowspill.pytorch.accelerator import resolve_device
from shadowspill.pytorch.task_inputs import prepare_inputs
from shadowspill.pytorch.toolchain import build_setup
from shadowspill.resources.description import resource_groups
from shadowspill.resources.preparation import prepare_resources

from .overlap_probe import measure_route, task_collective_count
from .session import MeasurementSession


def measure_overlap(
    profiled_program: ProfiledProgram,
    communication_info: CommunicationInfo,
    *,
    device: str | torch.device = "cuda",
    rank_map: Mapping[int, int] | None = None,
    stages: Sequence[int] | None = None,
    inputs: Mapping[str, Sequence[Any]] | None = None,
) -> Profiles:
    """Return a Profiles overlay; original timing/memory evidence stays intact.

    Call on every participant after communication calibration. Saved task forms
    must agree across workers. Merge with ``profiled_program.profiles``, then
    ``apply_profiles`` before planning. No build, runtime pool or training-step
    timing is used. Small/middle/large actual transfer sizes bound each probe.

    Communicating model tasks, insufficient free device memory and statistically
    unsupported samples are reported as missing evidence, never invented rates.
    Programs with model-owned communication resources are currently excluded:
    their opaque collective sequences cannot safely be interleaved by this probe.
    No sharded transfers means an empty overlay and needs no GPU/rendezvous.
    """
    program = profiled_program.program
    result = Profiles(setup=plain(program.setup))
    routes = overlap_routes(program, communication_info)
    if not routes or resource_groups(program.resources):
        result.metadata["overlap_calibration"] = {
            "status": "no_collective_transfers"
            if not routes
            else "model_owned_communication_not_probed",
            "measured_tasks": 0,
        }
        return result
    requests = collective_requests(program, None)
    target, failure = None, None
    try:
        target = resolve_device(device)
        assert target.index is not None
        require_compiled_device(program, target.index)
        if plain(program.setup) != build_setup(target):
            raise ValueError("overlap device/toolchain differs from the saved Program")
        if torch.cuda.memory.get_allocator_backend() != "native":
            raise ValueError(
                "overlap measurement requires the native PyTorch allocator"
            )
        count = 1 if program.parallelism is None else program.parallelism.world_size
        if not dist.is_initialized() or dist.get_world_size() != count:
            raise ValueError(
                "overlap measurement needs rendezvous at the Program world size"
            )
    except Exception as error:
        if not dist.is_initialized():
            raise
        failure = str(error)
    groups = GroupBindings(
        identity=digest(
            {
                "forms": [
                    (d.profile_key, d.compatibility_digest)
                    for d in program.task_definitions
                ],
                "setup": plain(program.setup),
                "routes": [(r.timing_class, r.bytes) for r in routes],
                "stages": None if stages is None else list(stages),
            }
        ),
        groups=[r.ranks for r in requests],
        rank_map=rank_map,
        topology=communication_info.topology,
        backend="nccl",
        startup_error=failure,
    )
    try:
        assert target is not None
        backend = shared_backend_setup(groups, requests, device_ordinal=target.index)
        expected = plain(communication_info.metadata.get("backend_setup", {}))
        # A calibration can contain more groups than this candidate uses.
        expected["groups"] = {
            k: expected.get("groups", {}).get(k) for k in backend["groups"]
        }
        if backend != expected:
            raise ValueError(
                "overlap backend differs from CommunicationInfo; "
                "recalibrate communication"
            )
        with (
            torch.cuda.device(target),
            torch.no_grad(),
            torch.random.fork_rng(devices=[target]),
            ExitStack() as cleanup,
        ):
            preparations = LoadedPreparations(target)
            cleanup.callback(preparations.close)
            cleanup.callback(torch.cuda.synchronize, target)
            preparations.add(program, tuple(program.executables))
            cleanup.enter_context(prepare_resources(program, target, rank_map))
            cleanup.callback(torch.cuda.synchronize, target)
            session = _OverlapSession(
                profiled_program, routes, target, inputs or {}, groups, preparations
            )
            for job in stage_jobs(program, stages):
                if "optimizer" in job or "gradient" in job:
                    component = job.get("optimizer", job.get("gradient"))
                    assert component is not None
                    key = component["profile_key"]
                    if key not in session.profiles.tasks:
                        torch.cuda.empty_cache()
                        arguments = prepare_inputs(
                            program, key, target, supplied=session.inputs.get(key)
                        )
                        session.measure(key, arguments)
                        del arguments
                else:
                    session.pair(job)
            result = session.profiles
            result.metadata["overlap_calibration"] = {
                "status": "complete",
                "source": "independent task/collective/paired trials; NCCL work events",
                "routes": [
                    {
                        "timing_class": r.timing_class,
                        "bytes": r.bytes,
                        "requests": [request.to_dict() for request in r.requests],
                    }
                    for r in routes
                ],
                "measured_tasks": sum(
                    bool(row.overlap_costs) for row in result.tasks.values()
                ),
                "attempted_tasks": len(result.tasks),
                "rank_aggregation": (
                    "minimum supported progress rate; missing ranks recorded"
                ),
                "limits": (
                    "pairwise NCCL interference; no host-copy interference "
                    "or model-owned collectives"
                ),
            }
            return result
    finally:
        groups.close()


class _OverlapSession(MeasurementSession):
    def __init__(
        self,
        profiled: ProfiledProgram,
        routes: tuple[OverlapRoute, ...],
        device: torch.device,
        inputs: Mapping[str, Sequence[Any]],
        groups: GroupBindings,
        preparations: LoadedPreparations,
    ) -> None:
        super().__init__(
            profiled.program,
            device,
            inputs,
            0,
            0,
            groups=groups,
            preparations=preparations,
        )
        self.baseline, self.routes = profiled.task_profiles, routes

    def measure(self, key: str, arguments: Sequence[Any]) -> None:
        if key in self.profiles.tasks:
            return
        assert self.groups is not None
        baseline = self.baseline[key]
        assert (
            baseline.duration_seconds is not None
            and baseline.workspace_bytes is not None
        )
        stream = self.streams[key]
        costs, evidence = [], []
        with torch.cuda.stream(stream):
            function = self.function(key)

            def invoke() -> Any:
                return function(*arguments)

            count, failure = 0, None
            try:
                count = task_collective_count(invoke, stream)
            except Exception as error:
                failure = str(error)
            self.groups.check(failure)
            communicating = any(
                self.groups.gather(bool(count or baseline.resource_phases))
            )
            if not communicating:
                for route in self.routes:
                    torch.cuda.empty_cache()
                    free, _ = torch.cuda.mem_get_info(self.device)
                    # Inputs are live already; admit one extra transfer buffer
                    # beside the previously observed complete task workspace.
                    spare = free - baseline.workspace_bytes - (32 << 20)
                    if min(self.groups.gather(spare)) < route.buffer_bytes:
                        evidence.append(
                            {
                                "timing_class": route.timing_class,
                                "bytes": route.bytes,
                                "status": "insufficient_device_memory",
                            }
                        )
                        continue
                    observation = measure_route(
                        invoke,
                        stream,
                        round(baseline.duration_seconds * 1e9),
                        route,
                        self.groups,
                    )
                    peers = self.groups.gather(observation)
                    supported = [
                        row["rates"] for row in peers if row["rates"] is not None
                    ]
                    if supported:
                        costs.append(
                            TaskOverlapCost(
                                route.timing_class,
                                route.bytes,
                                min(r[0] for r in supported),
                                min(r[1] for r in supported),
                            )
                        )
                    evidence.append(
                        {
                            "timing_class": route.timing_class,
                            "bytes": route.bytes,
                            "status": "measured"
                            if supported
                            else "unidentifiable_or_unstable",
                            "per_rank": peers,
                        }
                    )
        self.profiles.tasks[key] = TaskProfile(
            compatibility_digest=baseline.compatibility_digest,
            overlap_costs=tuple(costs),
            metadata={
                "overlap_calibration": {
                    "status": "task_contains_collectives"
                    if communicating
                    else "complete",
                    "observations": evidence,
                }
            },
        )
