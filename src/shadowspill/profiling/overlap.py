"""Enrich saved compute profiles with independently measured collective overlap."""

import os
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from time import perf_counter
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
from shadowspill.preparation.lifecycle import LoadedPreparations
from shadowspill.pytorch.accelerator import resolve_device
from shadowspill.pytorch.task_inputs import prepare_inputs
from shadowspill.pytorch.toolchain import build_setup
from shadowspill.resources.description import resource_groups
from shadowspill.resources.preparation import prepare_resources

from .cache_identity import task_identities
from .overlap_cache import OverlapProfileCache, observed_cost
from .overlap_probe import (
    ProbeBaselines,
    ProbeWorkspace,
    measure_route,
    task_collective_count,
)
from .session import MeasurementSession


def measure_overlap(
    profiled_program: ProfiledProgram,
    communication_info: CommunicationInfo,
    *,
    device: str | torch.device = "cuda",
    rank_map: Mapping[int, int] | None = None,
    stages: Sequence[int] | None = None,
    inputs: Mapping[str, Sequence[Any]] | None = None,
    trials: int | None = None,
    cache: OverlapProfileCache | None = None,
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
    Default sampling uses three trials when their predicted work fits 100 ms,
    otherwise one. ``trials`` overrides this with a positive fixed count. A
    single conditioned pair estimates progress, not statistical repeatability.
    Reuse one ``OverlapProfileCache`` across candidates to skip compatible paired
    measurements. Explicit task inputs disable cache reuse on every participant.
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
        if trials is not None and (type(trials) is not int or trials < 1):
            raise ValueError("overlap trials must be a positive integer or None")
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
                "trials": trials,
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
            prepared_at = perf_counter()
            preparations = LoadedPreparations(target)
            cleanup.callback(preparations.close)
            cleanup.callback(torch.cuda.synchronize, target)
            cleanup.enter_context(prepare_resources(program, target, rank_map))
            cleanup.callback(torch.cuda.synchronize, target)
            session = _OverlapSession(
                profiled_program,
                routes,
                target,
                inputs or {},
                groups,
                preparations,
                trials=trials,
                cache=cache,
                communication_reserve_bytes=communication_info.fixed_device_bytes or 0,
            )
            jobs = list(stage_jobs(program, stages))
            session.reuse_overlaps(jobs, backend)
            if session.cache_misses or cache is None or not session.overlap_cache:
                preparations.add(program, tuple(program.executables))
            prepared_seconds = perf_counter() - prepared_at
            jobs_at = perf_counter()
            for job in jobs:
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
            jobs_seconds = perf_counter() - jobs_at
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
                "sampling": {"trials": trials, "automatic_three_trial_budget_ms": 100},
                "cache": {
                    "hits": session.cache_hits,
                    "misses": session.cache_misses,
                },
                "wall_times_per_rank": groups.gather(
                    {
                        "preparation_seconds": prepared_seconds,
                        "jobs_seconds": jobs_seconds,
                        "probes_seconds": session.probe_seconds,
                        "inputs_and_dispatch_seconds": jobs_seconds
                        - session.probe_seconds,
                    }
                ),
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
        *,
        trials: int | None = None,
        cache: OverlapProfileCache | None = None,
        communication_reserve_bytes: int = 0,
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
        self.trials = trials
        self.probe_baselines = ProbeBaselines()
        self.probe_seconds = 0.0
        self.communication_reserve_bytes = communication_reserve_bytes
        self.overlap_cache = cache
        self.route_identities: dict[tuple[str, str, int], dict[str, Any]] = {}
        self.cached_routes: dict[tuple[str, str, int], dict[str, Any]] = {}

    def reuse_overlaps(
        self, jobs: Sequence[Mapping[str, Any]], backend: Mapping[str, Any]
    ) -> None:
        """Agree on identical evidence before any rank skips a collective."""
        assert self.groups is not None
        cache = self.overlap_cache
        if not all(self.groups.gather(cache is not None and not self.inputs)):
            self.groups.check(
                "required overlap cache unavailable or explicit inputs disable reuse"
                if cache is not None and cache.policy.require_hit
                else None
            )
            self.overlap_cache = None
            return
        assert cache is not None
        environment = {
            "allocator_config": os.environ.get(
                "PYTORCH_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
            ),
            "streams": self.streams.description,
            "rank_assignment": self.groups.assignment.logical_by_worker,
            "topology": self.groups.topology,
            "backend": plain(backend),
        }
        identities = task_identities(
            self.program, jobs, options=None, environment=environment
        )
        for key, identity in identities.items():
            task = {"digest": digest(identity), "environment": environment}
            for route in self.routes:
                self.route_identities[key, route.timing_class, route.bytes] = {
                    "protocol": "conditioned-shared-baselines/v1",
                    "task": task,
                    "route": {
                        "timing_class": route.timing_class,
                        "bytes": route.bytes,
                        "requests": [r.to_dict() for r in route.requests],
                    },
                    "trials": self.trials,
                    "automatic_three_trial_budget_ms": 100,
                }
        hits, failure = {}, None
        try:
            for item, identity in self.route_identities.items():
                if (record := cache.get(identity)) is not None:
                    hits[item] = record
        except Exception as error:
            failure = str(error)
        self.groups.check(failure)
        # A partial, stale or different per-node cache cannot cause ranks to
        # issue different probe sequences. Comparing evidence digests also
        # prevents rank-dependent planner costs after a concurrent refresh.
        accepted = set((item, digest(row)) for item, row in hits.items())
        accepted.intersection_update(*map(set, self.groups.gather(sorted(accepted))))
        self.cached_routes = {item: hits[item] for item, _ in accepted}
        failure = None
        try:
            for item, identity in self.route_identities.items():
                token = digest(identity)
                if item in self.cached_routes:
                    self.cache_hits.append(token)
                else:
                    self.cache_misses.append(token)
                    cache.policy.refuse_miss("overlap measurement", token)
        except Exception as error:
            failure = str(error)
        self.groups.check(failure)
        for key in identities:
            rows = [
                self.cached_routes.get((key, r.timing_class, r.bytes))
                for r in self.routes
            ]
            if all(row is not None for row in rows):
                self._record(key, [row for row in rows if row is not None], "cached")

    def _record(
        self,
        key: str,
        evidence: list[dict[str, Any]],
        status: str,
        wall: list[dict[str, float]] | None = None,
    ) -> None:
        self.profiles.tasks[key] = TaskProfile(
            compatibility_digest=self.baseline[key].compatibility_digest,
            overlap_costs=tuple(
                cost for row in evidence if (cost := observed_cost(row)) is not None
            ),
            metadata={
                "overlap_calibration": {
                    "status": status,
                    "observations": evidence,
                    "wall_times_per_rank": [] if wall is None else wall,
                }
            },
        )

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
        evidence = [
            self.cached_routes[key, route.timing_class, route.bytes]
            for route in self.routes
            if (key, route.timing_class, route.bytes) in self.cached_routes
        ]
        pending = [
            route
            for route in self.routes
            if (key, route.timing_class, route.bytes) not in self.cached_routes
        ]
        timing: dict[str, float] = {}
        started = perf_counter()
        with torch.cuda.stream(stream), ExitStack() as cleanup:
            function = self.function(key)
            timing["restore_seconds"] = perf_counter() - started

            def invoke() -> Any:
                return function(*arguments)

            communicating = any(self.groups.gather(bool(baseline.resource_phases)))
            if not communicating:
                before = perf_counter()
                # Release unrelated cached workspaces once, before conditioning.
                # Flushing after warmup makes the first timed task pay for CUDA
                # allocation/mapping again and biases the isolated baseline.
                torch.cuda.empty_cache()
                free, _ = torch.cuda.mem_get_info(self.device)
                # The first real collective can still initialize NCCL-owned
                # memory; task warmup can also retain declared lazy allocations.
                # Reserve their observed bounds before allocating the arena.
                retained = sum(baseline.metadata.get("warmup_persistent_bytes", ()))
                spare = min(
                    self.groups.gather(
                        free
                        - baseline.workspace_bytes
                        - retained
                        - self.communication_reserve_bytes
                        - (32 << 20)
                    )
                )
                admitted = []
                for route in pending:
                    if spare < route.buffer_bytes:
                        evidence.append(
                            {
                                "timing_class": route.timing_class,
                                "bytes": route.bytes,
                                "status": "insufficient_device_memory",
                            }
                        )
                    else:
                        admitted.append(route)
                timing["admission_seconds"] = perf_counter() - before
                if admitted:
                    before = perf_counter()
                    workspace, failure = None, None
                    try:
                        workspace = ProbeWorkspace(
                            max(route.buffer_bytes for route in admitted), stream
                        )
                        cleanup.callback(workspace.close)
                    except Exception as error:
                        failure = str(error)
                    self.groups.check(failure)
                    assert workspace is not None
                    timing["buffer_seconds"] = perf_counter() - before
                    before = perf_counter()
                    count, failure = 0, None
                    try:
                        count = task_collective_count(invoke, stream)
                    except Exception as error:
                        failure = str(error)
                    self.groups.check(failure)
                    communicating = any(self.groups.gather(bool(count)))
                    timing["warmup_seconds"] = perf_counter() - before
                for route in admitted if not communicating else ():
                    assert workspace is not None
                    before = perf_counter()
                    observation = measure_route(
                        invoke,
                        stream,
                        round(baseline.duration_seconds * 1e9),
                        route,
                        self.groups,
                        workspace=workspace,
                        baselines=self.probe_baselines,
                        task_key=key,
                        trials=self.trials,
                    )
                    peers = self.groups.gather(observation)
                    supported = [
                        row["rates"] for row in peers if row["rates"] is not None
                    ]
                    row = {
                        "timing_class": route.timing_class,
                        "bytes": route.bytes,
                        "status": "measured"
                        if supported
                        else "unidentifiable_or_unstable",
                        "per_rank": peers,
                    }
                    evidence.append(row)
                    failure = None
                    if self.overlap_cache is not None:
                        try:
                            self.overlap_cache.put(
                                self.route_identities[
                                    key, route.timing_class, route.bytes
                                ],
                                row,
                                persist=self.groups.logical_rank == 0,
                            )
                        except Exception as error:
                            failure = str(error)
                    self.groups.check(failure)
                    timing["routes_seconds"] = timing.get("routes_seconds", 0) + (
                        perf_counter() - before
                    )
        timing["total_seconds"] = perf_counter() - started
        self.probe_seconds += timing["total_seconds"]
        self._record(
            key,
            evidence,
            "task_contains_collectives" if communicating else "complete",
            self.groups.gather(timing),
        )
