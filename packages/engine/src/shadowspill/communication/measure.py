"""Public calibration orchestration, independent of execution pools and tasks."""

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import torch
import torch.distributed as dist

from shadowspill.artifacts._values import digest, plain
from shadowspill.artifacts.program import Program
from shadowspill.diagnostics.progress import event, phase
from shadowspill.memory import PinnedHostPool, SpillPool
from shadowspill.parallelism.agreement import agreement_digest
from shadowspill.pytorch.accelerator import resolve_device
from shadowspill.pytorch.toolchain import build_setup
from shadowspill.task.profiling import ProfilingOptions, resolve_options

from .aggregation import combine_host
from .binding.backend import shared_backend_setup
from .binding.groups import GroupBindings
from .binding.ranks import RankAssignment
from .host import measure_host_copies
from .info import CommunicationInfo
from .measure_collectives import measure_collectives
from .memory_session import CommunicationMemory
from .placement import bind_to_device, host_placement
from .requests import CollectiveRequest, collective_requests


def measure_communication(
    *,
    program: Program | None = None,
    rank_map: Mapping[int, int] | None = None,
    collectives: Sequence[Mapping[str, Any]] | None = None,
    device: str | torch.device = "cuda",
    sizes_bytes: Sequence[int] = (1024, 1 << 20, 16 << 20, 64 << 20),
    options: ProfilingOptions | None = None,
    warmup_seconds: float | None = None,
    measure_seconds: float | None = None,
    spill: SpillPool | None = None,
    numa_binding: bool = True,
) -> CommunicationInfo:
    """Measure spill copy rates, and the collectives a Program issues.

    Pass the result to `plan_program`. Called with no ``program``,
    ``collectives`` or ``rank_map``, it times this GPU's spill copies alone.
    Otherwise every worker calls it after the caller's process-group
    rendezvous, with the same complete ``rank_map``: the workers time their
    copies together, then each collective group in turn, and all receive the
    same result, with the slowest worker's rates, the largest latencies and
    every observation. Communicator memory is recorded apart from timing.

    By default the process first takes the host placement an execution worker
    takes: thread affinity narrowed to the device's NUMA node, and that node's
    memory preferred for pinned buffers, for the rest of the process. The
    result records it in ``metadata["host_placement"]``.

    Parameters
    ----------
    program : Program, optional
        A Program whose collectives to measure, built for this GPU and
        toolchain.
    rank_map : mapping of int to int, optional
        Launcher rank to logical rank, for every worker.
    collectives : sequence of mapping, optional
        Collectives to measure instead of the Program's, each with an
        ``operation`` (``"all_gather"``, ``"reduce_scatter"`` or
        ``"all_reduce"``), ordered logical ``ranks``, ``tensor_bytes`` and
        ``dtype``, and optionally ``inplace``. ``tensor_bytes`` is the whole
        gathered output or scattered input, or each rank's all-reduce tensor.
        An empty sequence times copies only.
    device : str or torch.device, default "cuda"
        This worker's device.
    sizes_bytes : sequence of int, default (1024, 1 << 20, 16 << 20, 64 << 20)
        Copy sizes, in bytes, from which latency and rate are fitted; at
        least two distinct positive sizes.
    options : ProfilingOptions, optional
        Timing bounds. By default ``ProfilingOptions()``.
    warmup_seconds : float, optional
        Shortcut for ``conditioning_seconds`` of ``options``; not with
        ``options``.
    measure_seconds : float, optional
        Shortcut for ``measurement_seconds`` of ``options``; not with
        ``options``.
    spill : SpillPool, optional
        The spill pool configuration execution will use. By default ordinary
        pinned host copies.
    numa_binding : bool, default True
        Whether to take an execution worker's host placement first. False keeps the
        caller's, for execution that also runs with ``numa_binding=False``.

    Returns
    -------
    CommunicationInfo
        The measured copy rates and latencies, collective measurements and
        communicator memory.

    Raises
    ------
    ValueError
        If ``sizes_bytes`` has fewer than two distinct positive sizes,
        ``options`` comes with a shortcut, ``rank_map`` is incomplete, the
        rendezvous world size differs from the Program's, or the Program was
        built for another GPU or toolchain.
    RuntimeError
        If PyTorch's native allocator is not in use.
    TypeError
        If ``numa_binding`` is not a bool, ``spill`` is not a `SpillPool`, or
        ``options`` is not `ProfilingOptions`.
    """
    coordinated = program is not None or collectives is not None or rank_map is not None
    distributed = coordinated and dist.is_initialized() and dist.get_world_size() > 1
    world = dist.get_world_size() if distributed else 1
    requests: tuple[CollectiveRequest, ...] = ()
    identity, failure = "", None
    bindings = None
    memory = None
    placement: dict[str, Any] = {"numa_binding": numa_binding}
    try:
        if not isinstance(numa_binding, bool):
            raise TypeError("numa_binding must be a bool")
        if spill is not None and not isinstance(spill, SpillPool):
            raise TypeError("spill must be a SpillPool configuration")
        sizes = tuple(sorted(set(sizes_bytes)))
        if len(sizes) < 2 or any(type(size) is not int or size <= 0 for size in sizes):
            raise ValueError(
                "sizes_bytes needs at least two distinct positive integers"
            )
        options = resolve_options(options, warmup_seconds, measure_seconds)
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
        # Before any buffer is pinned or any communicator thread is created.
        if numa_binding:
            unplaced = bind_to_device(target.index or 0)
            if unplaced is not None:
                placement["unplaced"] = unplaced
        placement.update(host_placement())
        event(
            f"Host placement: {placement['cpus']} CPUs on NUMA node(s) "
            f"{placement['numa_nodes']}"
            + (
                f"; not placed: {placement['unplaced']}"
                if "unplaced" in placement
                else ""
            )
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
                "profiling_options": options.to_dict(),
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
        assert options is not None  # Invalid policies fail before worker agreement.
        with torch.cuda.device(target), torch.no_grad():
            host, failure = None, None
            try:
                with phase("GPU and host transfer measurements"):
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
            # Each worker's own placement stays beside its own rates.
            host = replace(
                host, metadata={**host.metadata, "host_placement": placement}
            )
            result = combine_host(host, bindings)
            with phase(f"Collective measurements · {len(requests)} configurations"):
                measurements = measure_collectives(
                    requests,
                    device=target,
                    setup=setup,
                    options=options,
                    bindings=bindings,
                    backend_setup=backend,
                )
            with phase("Measure retained communication memory"):
                fixed, memory_metadata = (
                    memory.finish(bindings, requests, setup, backend_setup=backend)
                    if memory is not None
                    else (0, {"memory_kind": "static"})
                )
        return replace(
            result,
            measurements=measurements,
            fixed_device_bytes=fixed,
            metadata={
                **result.metadata,
                **memory_metadata,
                "backend_setup": backend,
                "profiling_options": options.to_dict(),
            },
        )
    finally:
        if bindings is not None:
            bindings.close()
