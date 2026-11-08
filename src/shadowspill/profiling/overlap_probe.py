"""Bounded saved-task/NCCL trials using actual collective work-event intervals."""

from collections import defaultdict
from collections.abc import Callable
from statistics import median
from typing import Any

import torch

from shadowspill.communication.binding.groups import GroupBindings
from shadowspill.communication.operations import launch_collective
from shadowspill.communication.overlap import OverlapRoute
from shadowspill.errors import ProfilingError

from .observation.native import _library
from .overlap_statistics import PairObservation, paired_progress, probe_work_counts


def task_collective_count(
    function: Callable[[], Any], stream: torch.cuda.Stream
) -> int:
    """Warm once and refuse concurrent calibration for communicating tasks."""
    _library()  # Register observation operators, without replacing the allocator.
    torch.ops.shadowspill._collective_profile_begin(4096)
    try:
        result = function()
        del result
        stream.synchronize()
    finally:
        record = torch.ops.shadowspill._collective_profile_end()
    if record["errors"]:
        raise ProfilingError("incomplete task collective observation")
    return len(record["records"])


class _Probe:
    def __init__(
        self, route: OverlapRoute, groups: GroupBindings, stream: torch.cuda.Stream
    ) -> None:
        self.stream = torch.cuda.Stream(device=stream.device)  # type: ignore[no-untyped-call]
        dtype = getattr(torch, route.requests[0].dtype.removeprefix("torch."))
        width = torch.empty((), dtype=dtype).element_size()
        self.buffer = torch.zeros(
            route.buffer_bytes // width, dtype=dtype, device=stream.device
        )
        self.operations = []
        local = tuple(r for r in route.requests if groups.logical_rank in r.ranks)
        shard = self.buffer
        for request in local:
            if request.dtype != route.requests[0].dtype:
                raise ValueError("compound collective route changes dtype")
            count = len(request.ranks)
            if request.operation in {"all_gather", "reduce_scatter"}:
                if request.tensor_bytes != route.buffer_bytes or not request.inplace:
                    raise ValueError(
                        "transfer probe requires the declared in-place buffer"
                    )
                size = self.buffer.numel() // count
                shard = self.buffer.narrow(
                    0, request.ranks.index(groups.logical_rank) * size, size
                )
                source, output = (
                    (shard, self.buffer)
                    if request.operation == "all_gather"
                    else (self.buffer, shard)
                )
            else:
                if shard.numel() * width != request.tensor_bytes:
                    raise ValueError("replica reduction geometry disagrees with shard")
                source = output = shard
            if count > 1:
                self.operations.append(
                    (
                        request.operation,
                        output,
                        source,
                        groups.for_members(request.ranks),
                    )
                )
        self.stream.wait_stream(stream)

    def enqueue(self, gate: torch.cuda.Event, copies: int) -> None:
        self.stream.wait_event(gate)
        with torch.cuda.stream(self.stream):
            for _ in range(copies):
                for operation, output, source, group in self.operations:
                    launch_collective(operation, output, source, group).wait()


def measure_route(
    function: Callable[[], Any],
    stream: torch.cuda.Stream,
    runtime_ns: int,
    route: OverlapRoute,
    groups: GroupBindings,
) -> dict[str, Any]:
    """Record evidence per rank. Caller decides how symmetric ranks combine it."""
    probe, failure = None, None
    try:
        probe = _Probe(route, groups, stream)
    except Exception as error:
        failure = str(error)
    groups.check(failure)
    assert probe is not None
    task_count, copies = probe_work_counts(
        runtime_ns, round(route.duration_seconds * 1e9)
    )
    counts = groups.gather((task_count, copies))
    task_count, copies = (max(value[i] for value in counts) for i in (0, 1))
    stream.synchronize()
    probe.stream.synchronize()

    def sample(mode: str) -> PairObservation:
        gate, begin, end = [torch.cuda.Event(enable_timing=True) for _ in range(3)]  # type: ignore[no-untyped-call]
        gate.record(stream)
        torch.ops.shadowspill._collective_profile_begin(
            len(probe.operations) * copies + 16, gate.cuda_event
        )
        try:
            if task_count > 1 and mode != "task":
                probe.enqueue(gate, copies)
            begin.record(stream)
            for _ in range(task_count):
                if mode != "transfer":
                    result = function()
                    del result  # Reuse one task workspace, even within a burst.
            end.record(stream)
            if task_count == 1 and mode != "task":
                probe.enqueue(gate, copies)
        finally:
            record = torch.ops.shadowspill._collective_profile_end()
        expected = 0 if mode == "task" else len(probe.operations) * copies
        if record["errors"] or len(record["records"]) != expected:
            raise ProfilingError(
                "incomplete or unexpected collective sequence in overlap probe"
            )
        events = record["records"]
        width = len(probe.operations)
        transfers = tuple(
            (events[i]["start_ms"] * 1e6, events[i + width - 1]["end_ms"] * 1e6)
            for i in range(0, len(events), max(1, width))
        )
        return PairObservation(
            gate.elapsed_time(begin) * 1e6, gate.elapsed_time(end) * 1e6, transfers
        )

    try:
        sample("transfer")  # Initialize communicator/events before paired trials.
        samples: dict[str, list[PairObservation]] = defaultdict(list)
        for repeat in range(5):
            modes = ("task", "transfer", "both")
            for mode in modes if repeat % 2 == 0 else tuple(reversed(modes)):
                samples[mode].append(sample(mode))
        rates = paired_progress(samples["task"], samples["transfer"], samples["both"])
        return {
            "logical_rank": groups.logical_rank,
            "rates": rates,
            "status": "measured" if rates is not None else "unidentifiable_or_unstable",
            "task_count": task_count,
            "copies": copies,
            "trials": {
                mode: [_summary(row) for row in rows] for mode, rows in samples.items()
            },
        }
    finally:
        stream.synchronize()
        probe.stream.synchronize()


def _summary(row: PairObservation) -> dict[str, Any]:
    """Keep audit statistics, not thousands of raw event records per task."""
    interior = [
        end - start
        for start, end in row.transfers[1:]
        if start >= row.task_start_ns and end <= row.task_end_ns
    ]
    transfers = [end - start for start, end in row.transfers[1:]]
    return {
        "task_ns": row.task_end_ns - row.task_start_ns,
        "overlap_ns": sum(
            max(0.0, min(end, row.task_end_ns) - max(start, row.task_start_ns))
            for start, end in row.transfers
        ),
        "interior_transfers": len(interior),
        "interior_median_ns": median(interior) if interior else None,
        "transfer_median_ns": median(transfers) if transfers else None,
    }
