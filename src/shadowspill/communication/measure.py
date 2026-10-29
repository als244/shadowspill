"""Public calibration orchestration, independent of execution pools and tasks."""

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import torch
import torch.distributed as dist

from shadowspill.artifacts._values import digest, plain
from shadowspill.artifacts.program import Program
from shadowspill.memory import PinnedHostPool, SpillPool
from shadowspill.parallelism.agreement import agreement_digest
from shadowspill.pytorch.accelerator import resolve_device
from shadowspill.pytorch.toolchain import build_setup
from shadowspill.task.profiling import ProfilingOptions

from .aggregation import combine_host
from .binding.backend import shared_backend_setup
from .binding.groups import GroupBindings
from .binding.ranks import RankAssignment
from .host import measure_host_copies
from .info import CommunicationInfo
from .measure_collectives import measure_collectives
from .memory_session import CommunicationMemory
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
    spill: SpillPool | None = None,
) -> CommunicationInfo:
    """Calibrate spill copies and optionally a Program's lowered collectives.

    Without Program/overrides/rank_map, measure only this GPU's host copies.
    Distributed calls run on every worker after caller-owned rendezvous, with
    the same full worker-to-logical rank_map. Workers concurrently measure host
    copies, then each requested group separately. Every worker receives the same
    result. Collective allocation evidence and retained communicator bytes are
    recorded separately from timing. NCCL-owned temporary allocations use the
    worker's external headroom; execution warms groups before memory admission.

    collectives replaces derived requests; [] selects host copies only. Each
    entry has operation, ordered logical ranks, tensor_bytes and dtype. Bytes
    mean full gathered output/scatter input, or each all-reduce tensor.
    spill selects an existing pool configuration; None uses ordinary pinned-host
    copies. External backends use bounded native probe pools without installing
    the execution allocator. Build and compute profiling need no spill setting.
    """
    coordinated = program is not None or collectives is not None or rank_map is not None
    distributed = coordinated and dist.is_initialized() and dist.get_world_size() > 1
    world = dist.get_world_size() if distributed else 1
    requests: tuple[CollectiveRequest, ...] = ()
    identity, failure = "", None
    bindings = None
    memory = None
    try:
        if spill is not None and not isinstance(spill, SpillPool):
            raise TypeError("spill must be a SpillPool configuration")
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
        if requests:
            memory = CommunicationMemory(target)
        # What the workers must agree on: the exchange, not each one's own
        # build, so Programs built for different hardware calibrate together.
        identity = digest(
            {
                "program": None if program is None else agreement_digest(program),
                "requests": [item.to_dict() for item in requests],
                "sizes": sizes,
                "warmup": warmup_seconds,
                "measure": measure_seconds,
                "spill_kind": "pinned_host" if spill is None else spill.kind_name,
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
        backend = shared_backend_setup(bindings, requests, device_ordinal=target.index)
        with torch.cuda.device(target), torch.no_grad():
            host, failure = None, None
            try:
                if spill is None or type(spill) is PinnedHostPool:
                    host = measure_host_copies(
                        device=target,
                        sizes=sizes,
                        options=options,
                        bindings=bindings,
                    )
                else:
                    from .spill import measure_spill

                    host = measure_spill(
                        spill,
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
                backend_setup=backend,
            )
            fixed, memory_metadata = (
                memory.finish(bindings, requests, setup, backend_setup=backend)
                if memory is not None
                else (0, {"memory_kind": "static"})
            )
        return replace(
            result,
            measurements=measurements,
            fixed_device_bytes=fixed,
            metadata={**result.metadata, **memory_metadata, "backend_setup": backend},
        )
    finally:
        if bindings is not None:
            bindings.close()
