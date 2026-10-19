"""Independent paired measurements of compute and sharded collective progress.

No training-step timings enter this calibration. Only positive progress rates
supported by repeated actual overlap become simulator costs. In particular,
delays after a task ends are not attributed to compute interference.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from shadowspill.errors import ProfilingError
from shadowspill.ir import TaskOverlapCost
from shadowspill.pytorch.distributed import current
from shadowspill.pytorch.distributed._backing_storage import collective_timing_class
from shadowspill.pytorch.distributed._profiling import invocation_batch

if TYPE_CHECKING:
    from shadowspill.pytorch.distributed import BoundDistributed
    from shadowspill.runtime.telemetry import TaskWorkspaceProfile

    from .boundary import AllocatorBoundary
    from .measurement import MeasuredTask


@dataclass(frozen=True)
class PairObservation:
    task_start_ns: float
    task_end_ns: float
    transfers: tuple[tuple[float, float], ...]


def paired_progress(
    solo_tasks: Sequence[PairObservation],
    solo_transfers: Sequence[PairObservation],
    together: Sequence[PairObservation],
) -> tuple[float, float] | None:
    """Estimate lost task progress and per-copy progress within that task.

    The first transfer can include rank-arrival skew and is excluded from the
    per-copy comparison. Transfers straddling the task's end are excluded too:
    their tail can wait for another rank after local compute has finished.
    Invalid or unidentifiable observations are rejected, never floored to an
    arbitrarily tiny progress rate. At least three paired trials must agree.
    """
    task_rates, transfer_rates = [], []
    for task, copies, pair in zip(solo_tasks, solo_transfers, together, strict=True):
        solo = task.task_end_ns - task.task_start_ns
        elapsed = pair.task_end_ns - pair.task_start_ns
        overlap = sum(
            max(0.0, min(end, pair.task_end_ns) - max(start, pair.task_start_ns))
            for start, end in pair.transfers
        )
        if solo <= 0 or overlap <= 0 or overlap > elapsed + 1000:
            continue
        task_rate = 1 - (elapsed - solo) / overlap
        reference = [end - start for start, end in copies.transfers[1:]]
        interior = [
            end - start
            for start, end in pair.transfers[1:]
            if start >= pair.task_start_ns and end <= pair.task_end_ns
        ]
        if not reference or not interior or min(reference + interior) <= 0:
            continue
        transfer_rate = median(reference) / median(interior)
        if not 0 < task_rate <= 1 or not 0 < transfer_rate <= 1:
            continue
        task_rates.append(task_rate)
        transfer_rates.append(transfer_rate)
    if len(task_rates) < 3:
        return None
    for values in (task_rates, transfer_rates):
        center = median(values)
        if median(abs(value - center) for value in values) > 0.25 * center:
            return None
    return median(task_rates), median(transfer_rates)


@dataclass(frozen=True)
class _Route:
    timing_class: str
    full_bytes: int
    bytes: int
    solo_ns: int
    operation: str
    dtype: torch.dtype
    group: Any
    replica_group: Any


def _probe_work_counts(runtime_ns: int, transfer_ns: int) -> tuple[int, int]:
    """Cover several large transfers without retaining multiple task workspaces.

    A single short task may finish before even the second collective does.
    Repeated calls provide interior copies whose progress is identifiable.
    Bounds keep tiny tasks and messages from making calibration unbounded.
    """
    tasks = max(1, min(8, math.ceil(4 * transfer_ns / max(1, runtime_ns))))
    copies = max(3, min(128, math.ceil(2 * tasks * runtime_ns / transfer_ns)))
    return tasks, copies


def _routes(bound: BoundDistributed) -> tuple[_Route, ...]:
    groups = {
        bound.groups.source_names[group.group_name]: group
        for group in bound.groups.groups.values()
    }
    by_class: dict[str, list[_Route]] = defaultdict(list)
    for key, record in bound.transfer_calibration.items():
        alias, size = key[:2]
        group = groups[alias]
        count = dist.get_world_size(group)
        if count == 1 or not record["duration_ns"]:
            continue
        operation = "gather" if len(key) == 2 else "reduce"
        dtype = (
            torch.uint8
            if operation == "gather"
            else getattr(torch, key[2].removeprefix("torch."))
        )
        replicas = None if len(key) == 2 or key[3] is None else groups[key[3]]
        timing_class = collective_timing_class(
            operation,
            count,
            dtype,
            1 if replicas is None else dist.get_world_size(replicas),
        )
        by_class[timing_class].append(
            _Route(
                timing_class,
                size,
                size - size // count,
                record["duration_ns"],
                operation,
                dtype,
                group,
                replicas,
            )
        )
    result: list[_Route] = []
    for _, routes in sorted(by_class.items()):
        sizes = sorted({route.full_bytes for route in routes})
        # Small, middle and largest actual execution objects; no synthetic
        # size is substituted for an object used by this program.
        selected = {sizes[0], sizes[len(sizes) // 2], sizes[-1]}
        result.extend(route for route in routes if route.full_bytes in selected)
    return tuple(result)


def measured_overlap_costs(
    boundary: AllocatorBoundary,
    source: MeasuredTask,
    stream: torch.cuda.Stream,
    runtime_ns: int,
    workspace: TaskWorkspaceProfile,
    *,
    task_has_collectives: bool,
) -> tuple[TaskOverlapCost, ...]:
    """Probe the exact task against the program's borrowed sharding groups.

    Collective-bearing tasks are excluded: their ordered internal phases model
    the communicator dependency, and concurrent probes must not reorder those
    collectives. Extra buffers must fit beside the measured task workspace.
    """
    bound = current()
    if bound is None or not bound.transfer_sharding:
        return ()
    if any(bound.control.exchange("overlap/task-collectives", task_has_collectives)):
        return ()
    routes = _routes(bound)
    signature = [(r.timing_class, r.full_bytes) for r in routes]
    bound.control.agree("overlap/routes", signature)
    pool = boundary.statistics().allocator_pool
    spare = min(
        int(pool.free_bytes) - workspace.peak_charged_bytes,
        int(pool.largest_free_range_bytes)
        - max(workspace.peak_extent_bytes, default=0),
    ) - (2 << 20)
    spare = min(bound.control.exchange("overlap/spare-bytes", spare))
    measured: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    for route in routes:
        if route.full_bytes > spare:
            continue
        cost = _measure_route(boundary, source, stream, runtime_ns, route)
        peers = bound.control.exchange("overlap/progress", cost)
        # Every symmetric rank needs the same table shape. Use supported
        # peer measurements conservatively, without fabricating missing data.
        supported = [value for value in peers if value is not None]
        if supported:
            measured[route.timing_class, route.bytes].append(
                (min(v[0] for v in supported), min(v[1] for v in supported))
            )
    return tuple(
        TaskOverlapCost(name, size, min(v[0] for v in rates), min(v[1] for v in rates))
        for (name, size), rates in sorted(measured.items())
    )


def _measure_route(
    boundary: AllocatorBoundary,
    source: MeasuredTask,
    stream: torch.cuda.Stream,
    runtime_ns: int,
    route: _Route,
) -> tuple[float, float] | None:
    width = torch.empty((), dtype=route.dtype).element_size()
    buffers = [
        torch.zeros(route.full_bytes // width, dtype=route.dtype, device=stream.device)
    ]
    count = dist.get_world_size(route.group)
    buffers.append(
        buffers[0].narrow(
            0,
            dist.get_rank(route.group) * (buffers[0].numel() // count),
            buffers[0].numel() // count,
        )
    )
    communication = torch.cuda.Stream(device=stream.device)  # type: ignore[no-untyped-call]
    stream.synchronize()
    bound = current()
    assert bound is not None
    task_count, copies = _probe_work_counts(runtime_ns, route.solo_ns)
    counts = bound.control.exchange("overlap/work-counts", (task_count, copies))
    task_count, copies = (max(value[i] for value in counts) for i in (0, 1))

    def enqueue_copies(gate: torch.cuda.Event) -> None:
        communication.wait_event(gate)
        with torch.cuda.stream(communication):
            for _ in range(copies):
                if route.operation == "gather":
                    dist.all_gather_into_tensor(
                        buffers[0], buffers[1], group=route.group, async_op=True
                    ).wait()
                else:
                    dist.reduce_scatter_tensor(
                        buffers[1], buffers[0], group=route.group, async_op=True
                    ).wait()
                    if route.replica_group is not None:
                        dist.all_reduce(
                            buffers[1], group=route.replica_group, async_op=True
                        ).wait()

    def sample(mode: str) -> PairObservation:
        gate, begin, end = [torch.cuda.Event(enable_timing=True) for _ in range(3)]  # type: ignore[no-untyped-call]
        with invocation_batch():
            gate.record(stream)
            torch.ops.shadowspill._collective_profile_begin(
                2 * copies + 16, gate.cuda_event
            )
            try:
                # Large copies begin before the bounded task sequence. Each
                # task drains its own allocator scope, so the probe never
                # accumulates task workspaces. Small-message probes retain
                # compute-first launch: posting 128 tiny copies up front can
                # otherwise finish all of them before the task is submitted.
                if task_count > 1 and mode != "task":
                    enqueue_copies(gate)
                for index in range(task_count):
                    with source.open() as task, boundary.scope(stream):
                        if index == 0:
                            begin.record(stream)
                        if mode != "transfer":
                            result = task()
                            del result
                        if index + 1 == task_count:
                            end.record(stream)
                        if task_count == 1 and mode != "task":
                            enqueue_copies(gate)
            finally:
                record = torch.ops.shadowspill._collective_profile_end()
        boundary.require_idle(problem="task collective overlap sample")
        if record["errors"]:
            raise ProfilingError("incomplete events in collective overlap calibration")
        events = record["records"]
        width = 2 if route.replica_group is not None else 1
        expected = 0 if mode == "task" else width * copies
        if len(events) != expected:
            raise ProfilingError("unexpected collective sequence in overlap probe")
        transfers = tuple(
            (events[i]["start_ms"] * 1e6, events[i + width - 1]["end_ms"] * 1e6)
            for i in range(0, len(events), width)
        )
        return PairObservation(
            gate.elapsed_time(begin) * 1e6, gate.elapsed_time(end) * 1e6, transfers
        )

    try:
        sample("transfer")
        samples: dict[str, list[PairObservation]] = defaultdict(list)
        for repeat in range(5):
            modes: tuple[str, ...] = ("task", "transfer", "both")
            if repeat % 2:
                modes = tuple(reversed(modes))
            for mode in modes:
                samples[mode].append(sample(mode))
        return paired_progress(samples["task"], samples["transfer"], samples["both"])
    finally:
        buffers.clear()
        boundary.drain(stream, problem="collective overlap buffer cleanup")
