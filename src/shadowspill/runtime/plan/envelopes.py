"""Per-task allocation limits derived from validated measurement evidence."""

from __future__ import annotations

from shadowspill.runtime.plan.common import TaskMemoryEnvelope
from shadowspill.task.allocations import TaskAllocationOperation
from shadowspill.task.profiles import TaskMeasurement


def task_memory_envelope(
    measurement: TaskMeasurement,
    *,
    retained_output_leaves: tuple[int, ...] = (),
    minimum_scratch_reserve_bytes: int = 0,
) -> TaskMemoryEnvelope:
    live: dict[int, tuple[int, int]] = {}
    live_requested = 0
    live_charged = 0
    peak_requested = 0
    peak_charged = 0
    maximum_requested = 0
    maximum_charged = 0
    for event in measurement.allocation_trace:
        sizes = event.requested_bytes, event.charged_bytes
        if event.operation is TaskAllocationOperation.ALLOCATE:
            live[event.allocation_ordinal] = sizes
            live_requested += sizes[0]
            live_charged += sizes[1]
            peak_requested = max(peak_requested, live_requested)
            peak_charged = max(peak_charged, live_charged)
            maximum_requested = max(maximum_requested, sizes[0])
            maximum_charged = max(maximum_charged, sizes[1])
        else:
            prior = live.pop(event.allocation_ordinal)
            live_requested -= prior[0]
            live_charged -= prior[1]
    allocation_contract = measurement.allocation_contract
    if allocation_contract is not None:
        allocation_contract = allocation_contract.for_retained_output_leaves(
            retained_output_leaves
        )
    scratch_peak_requested = max(
        measurement.dynamic_scratch_peak_requested_bytes,
        minimum_scratch_reserve_bytes,
    )
    scratch_peak_charged = max(
        measurement.dynamic_scratch_peak_charged_bytes,
        minimum_scratch_reserve_bytes,
    )
    return TaskMemoryEnvelope(
        maximum_requested_allocation_bytes=max(
            maximum_requested,
            measurement.dynamic_scratch_maximum_requested_bytes,
            minimum_scratch_reserve_bytes,
        ),
        maximum_charged_allocation_bytes=max(
            maximum_charged,
            measurement.dynamic_scratch_maximum_charged_bytes,
            minimum_scratch_reserve_bytes,
        ),
        live_requested_allocation_limit_bytes=_envelope_limit(
            peak_requested + scratch_peak_requested
        ),
        live_charged_allocation_limit_bytes=_envelope_limit(
            peak_charged + scratch_peak_charged
        ),
        dynamic_scratch_maximum_allocation_bytes=max(
            measurement.dynamic_scratch_maximum_charged_bytes,
            minimum_scratch_reserve_bytes,
        ),
        dynamic_scratch_live_limit_bytes=_envelope_limit(scratch_peak_charged),
        allocation_path_digests=tuple(
            item.compatibility_digest
            for item in measurement.allocation_path_observations
        ),
        allocation_contract=allocation_contract,
    )


def _envelope_limit(profiled_bytes: int) -> int:
    if profiled_bytes == 0:
        return 0
    two_mib = 2 << 20
    with_headroom = (profiled_bytes * 5 + 3) // 4
    return ((with_headroom + two_mib - 1) // two_mib) * two_mib
