"""Drain the traced invocation and decode allocator/runtime observations."""

from dataclasses import dataclass

from shadowspill.diagnostics.step import AllocatorTrace, RuntimeTrace
from shadowspill.diagnostics.timing import ArmedExecutionTiming
from shadowspill.runtime.abi import AdapterStatistics
from shadowspill.runtime.plan.bridge import RuntimeBridge
from shadowspill.runtime.plan.tracing import end_and_read_runtime_trace, statistics
from shadowspill.runtime.trace import CapturedRuntimeTrace


@dataclass(frozen=True, slots=True)
class TraceEvidence:
    runtime_trace: CapturedRuntimeTrace
    statistics_before: AdapterStatistics
    statistics_after: AdapterStatistics


def validate_completed_timing(timing: ArmedExecutionTiming) -> None:
    if not timing.started:
        raise RuntimeError("no execution timing measurement has started")
    if not timing.finished:
        raise RuntimeError("the execution timing measurement has not finished")
    if timing.stream is None:
        raise RuntimeError("execution timing has no compute stream")
    if timing.statistics_before is None:
        raise RuntimeError("execution timing omitted its initial statistics")


def resolve_trace_evidence(
    timing: ArmedExecutionTiming,
    bridge: RuntimeBridge,
) -> TraceEvidence:
    """Drain the explicitly traced call before copying its evidence."""

    if timing.stream is None:
        raise AssertionError("validated execution timing lost its compute stream")
    # Everything measured was recorded on this device, so waiting for the
    # device is waiting for all of it. Only a frontend can wait.
    runtime = bridge.runtime
    runtime.frontend.synchronize(int(runtime._installed.admission.device_ordinal))
    # Terminal transfers must be included in the same invocation trace.
    bridge.wait_runtime_idle()
    if timing.collect_collectives is not None:
        timing.collectives = timing.collect_collectives()
        timing.collect_collectives = None
    runtime_trace = end_and_read_runtime_trace(bridge)
    statistics_before = timing.statistics_before
    if statistics_before is None:
        raise AssertionError("validated execution timing lost initial statistics")
    return TraceEvidence(
        runtime_trace=runtime_trace,
        statistics_before=statistics_before,
        statistics_after=statistics(bridge),
    )


def build_allocator_trace(evidence: TraceEvidence) -> AllocatorTrace:
    # The allocator's own pool answers for occupancy and fragmentation; the
    # runtime answers for the event ring.
    before = evidence.statistics_before.allocator_pool
    after = evidence.statistics_after.allocator_pool
    return AllocatorTrace(
        events=evidence.runtime_trace.allocation_events,
        live_allocations_before=int(before.live_allocations),
        live_allocations_after=int(after.live_allocations),
        allocated_bytes_before=int(before.allocated_bytes),
        allocated_bytes_after=int(after.allocated_bytes),
        peak_allocated_bytes=int(after.peak_allocated_bytes),
        free_bytes_after=int(after.free_bytes),
        free_prefix_bytes_after=int(after.free_prefix_bytes),
        largest_free_range_bytes_after=int(after.largest_free_range_bytes),
        external_fragmentation_bytes_after=int(after.external_fragmentation_bytes),
        blocked_allocators_after=int(after.blocked_allocators),
        overflow=bool(evidence.statistics_after.runtime.allocation_event_overflow),
    )


def build_runtime_trace(evidence: TraceEvidence) -> RuntimeTrace:
    before = evidence.statistics_before
    after = evidence.statistics_after
    allocation_requests = int(after.allocation_callbacks - before.allocation_callbacks)
    zero_byte_requests = int(
        after.zero_size_allocation_callbacks - before.zero_size_allocation_callbacks
    )
    runtime_trace = evidence.runtime_trace
    return RuntimeTrace(
        wait_events_inserted=int(
            after.runtime.wait_events_inserted - before.runtime.wait_events_inserted
        ),
        allocation_requests=allocation_requests,
        zero_byte_allocation_requests=zero_byte_requests,
        materialized_allocation_requests=allocation_requests - zero_byte_requests,
        free_requests=int(after.free_callbacks - before.free_callbacks),
        record_stream_callbacks=int(
            after.record_stream_callbacks - before.record_stream_callbacks
        ),
        event_queries=int(after.backend.event_queries - before.backend.event_queries),
        queued_actions_after=int(after.runtime.queued_actions),
        pending_retirements_after=int(after.runtime.pending_retirements),
        callback_failures_after=int(after.callback_failures),
        step_id=runtime_trace.step_id,
        began_at_ns=runtime_trace.began_at_ns,
        ended_at_ns=runtime_trace.ended_at_ns,
        event_capacity=runtime_trace.event_capacity,
        allocation_event_capacity=runtime_trace.allocation_event_capacity,
        event_overflow=runtime_trace.event_overflow,
        allocation_event_overflow=runtime_trace.allocation_event_overflow,
        events=runtime_trace.events,
    )
