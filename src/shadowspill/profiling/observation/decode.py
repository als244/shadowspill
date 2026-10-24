"""Convert native requests to pointer-free lifetimes with target charging."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from shadowspill.errors import CaptureError
from shadowspill.task.allocations import TaskAllocationEvent, TaskAllocationOperation
from shadowspill.task.layout import _allocation_live_set_peak
from shadowspill.task.profiles import TaskOutputInputBinding


@dataclass(frozen=True)
class AllocationObservation:
    trace: tuple[TaskAllocationEvent, ...]
    input_bindings: tuple[TaskOutputInputBinding, ...]
    persistent_bytes: tuple[int, ...]
    off_device_outputs: tuple[int, ...]

    @property
    def workspace_bytes(self) -> int:
        return _allocation_live_set_peak(self.trace, set())

    @property
    def workspace_requested_bytes(self) -> int:
        requested = tuple(
            replace(event, charged_bytes=event.requested_bytes) for event in self.trace
        )
        return _allocation_live_set_peak(requested, set())


def decode_allocations(
    events: Sequence[Mapping[str, Any]],
    output_views: Mapping[int, Sequence[tuple[int, int]]],
    input_bindings: Sequence[TaskOutputInputBinding],
    off_device_outputs: Sequence[int] = (),
    *,
    alignment: int = 256,
) -> AllocationObservation:
    if alignment <= 0:
        raise ValueError("allocation alignment must be positive")
    trace: list[TaskAllocationEvent] = []
    latest: dict[int, int] = {}
    live: dict[int, TaskAllocationEvent] = {}
    allocation_positions: dict[int, int] = {}
    for event in events:
        if event["pool_first"] or event["pool_second"]:
            raise CaptureError(
                "native profiling does not yet support private memory pools"
            )
        action, address = event["action"], event["address"]
        if action == "out_of_memory":
            raise CaptureError("allocation observation contains an out-of-memory event")
        if action == "allocate":
            if address in live:
                raise CaptureError(
                    "native allocation reused an address before retirement"
                )
            ordinal = len(allocation_positions)
            size = event["requested_bytes"]
            # Runtime extents retain the exact request. Address alignment is a
            # separate placement constraint, not per-allocation size rounding.
            charged = size
            value = TaskAllocationEvent(
                ordinal,
                TaskAllocationOperation.ALLOCATE,
                size,
                charged,
                alignment_bytes=alignment,
            )
            latest[address] = ordinal
            live[address] = value
            allocation_positions[ordinal] = len(trace)
            trace.append(value)
        elif action == "free_completed" and address in live:
            value = live.pop(address)
            trace.append(replace(value, operation=TaskAllocationOperation.FREE))
    for address, views in output_views.items():
        if address not in latest:
            raise CaptureError(
                "output storage was neither allocated by this task nor an input"
            )
        position = allocation_positions[latest[address]]
        value = trace[position]
        if any(offset < 0 or offset > value.requested_bytes for _, offset in views):
            raise CaptureError("output view extends beyond its allocation request")
        trace[position] = replace(
            value,
            output_leaf_indices=tuple(v[0] for v in views),
            output_view_offsets=tuple(v[1] for v in views),
        )
    persistent = tuple(
        value.charged_bytes
        for address, value in live.items()
        if address not in output_views
    )
    return AllocationObservation(
        tuple(trace), tuple(input_bindings), persistent, tuple(off_device_outputs)
    )
