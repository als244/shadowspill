"""Public calibration orchestration, independent of execution pools and tasks."""

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import torch
import torch.distributed as dist

from shadowspill.artifacts._values import digest, plain
from shadowspill.artifacts.program import Program
from shadowspill.pytorch.accelerator import resolve_device
from shadowspill.pytorch.toolchain import build_setup
from shadowspill.task.profiling import ProfilingOptions

from .aggregation import combine_host
from .binding.groups import GroupBindings
from .binding.ranks import RankAssignment
from .host import measure_host_copies
from .info import CommunicationInfo
from .measure_collectives import measure_collectives
from .requests import CollectiveRequest, collective_requests


def measure_communication(
    *,
    program: Program | None = None,
    rank_map: Mapping[int, int] | None = None,
    collectives: Sequence[Mapping[str, Any]] | None = None,
    device: str | torch.device = "cuda",
    sizes_bytes: Sequence[int] = (1024, 1 << 20, 16 << 20, 64 << 20),
    warmup_seconds: float = 0.05,
    measure_seconds: float = 0.18,
) -> CommunicationInfo:
    """Calibrate copies and optionally a Program's lowered collectives.

    Without Program/overrides/rank_map, measure only this GPU's host copies.
    Distributed calls run on every worker after caller-owned rendezvous, with
    the same full worker-to-logical rank_map. Workers concurrently measure host
    copies, then each requested group separately. Every worker receives the same
    result. Exact collective observations do not imply a general interconnect
    bandwidth or qualify communicator memory for execution.

    collectives replaces derived requests; [] selects host copies only. Each
    entry has operation, ordered logical ranks, tensor_bytes and dtype. Bytes
    mean full gathered output/scatter input, or each all-reduce tensor.
    """
    coordinated = program is not None or collectives is not None or rank_map is not None
    distributed = coordinated and dist.is_initialized() and dist.get_world_size() > 1
    world = dist.get_world_size() if distributed else 1
    requests: tuple[CollectiveRequest, ...] = ()
    identity, failure = "", None
    bindings = None
    try:
        sizes = tuple(sorted(set(sizes_bytes)))
        if len(sizes) < 2 or any(type(size) is not int or size <= 0 for size in sizes):
            raise ValueError(
                "sizes_bytes needs at least two distinct positive integers"
            )
        options = ProfilingOptions(
            conditioning_seconds=warmup_seconds,
            conditioning_wall_seconds=max(1.0, warmup_seconds * 4),
            measurement_seconds=measure_seconds,
            measurement_wall_seconds=max(1.0, measure_seconds * 4),
        )
        requests = collective_requests(program, collectives)
        if program is not None:
            count = 1 if program.parallelism is None else program.parallelism.world_size
            if world != count:
                raise ValueError(
                    "initialize rendezvous with the Program's declared world size"
                )
        if not distributed:
            assignment = RankAssignment.resolve(1, rank_map)
            for request in requests:
                assignment.workers(request.ranks)
        target = resolve_device(device)
        if torch.cuda.memory.get_allocator_backend() != "native":
            raise RuntimeError(
                "communication measurement requires the native PyTorch allocator"
            )
        setup = build_setup(target)
        if program is not None and program.setup and plain(program.setup) != setup:
            raise ValueError(
                "Program build setup does not match this calibration GPU/runtime"
            )
        identity = digest(
            {
                "program": None if program is None else program.digest,
                "requests": [item.to_dict() for item in requests],
                "sizes": sizes,
                "warmup": warmup_seconds,
                "measure": measure_seconds,
                "setup": setup,
            }
        )
    except Exception as error:
        if not distributed:
            raise
        failure = str(error)
    try:
        if distributed:
            bindings = GroupBindings(
                identity=identity,
                groups=[request.ranks for request in requests],
                rank_map=rank_map,
                topology={},
                backend="nccl",
                startup_error=failure,
            )
        with torch.cuda.device(target), torch.no_grad():
            host, failure = None, None
            try:
                host = measure_host_copies(
                    device=target,
                    sizes=sizes,
                    options=options,
                    bindings=bindings,
                )
            except Exception as error:
                if bindings is None:
                    raise
                failure = str(error)
            if bindings is not None:
                bindings.check(failure)
            assert host is not None
            result = combine_host(host, bindings)
            measurements = measure_collectives(
                requests,
                device=target,
                setup=setup,
                options=options,
                bindings=bindings,
            )
        return replace(result, measurements=measurements)
    finally:
        if bindings is not None:
            bindings.close()
