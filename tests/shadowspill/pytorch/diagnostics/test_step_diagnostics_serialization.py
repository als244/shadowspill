from __future__ import annotations

import io

import torch

from shadowspill.planner.diagnostics.mapping import FrozenMapping
from shadowspill.pytorch.diagnostics.execution import (
    AllocatorTrace,
    ExecutionTiming,
    RuntimeTrace,
    StepDiagnostics,
    StepTimingSummary,
    TransferTrace,
)


def _diagnostics() -> StepDiagnostics:
    timing = ExecutionTiming(
        compute_seconds=1.0,
        optimizer_seconds=0.1,
        dispatch_call_seconds=1.1,
        dispatch_startup_wait_seconds=0.0,
        dispatch_initial_actions_seconds=0.0,
        trace_setup_seconds=0.0,
        phase_gpu_seconds=(("forward", 1.0),),
        tasks=FrozenMapping({}),
    )
    allocator = AllocatorTrace(
        events=(),
        live_allocations_before=0,
        live_allocations_after=0,
        allocated_bytes_before=0,
        allocated_bytes_after=0,
        peak_allocated_bytes=0,
        free_bytes_after=1,
        free_prefix_bytes_after=1,
        largest_free_range_bytes_after=1,
        external_fragmentation_bytes_after=0,
        blocked_allocators_after=0,
        overflow=False,
    )
    transfers = TransferTrace(
        actions=(),
        fetch_transfers=0,
        evict_transfers=0,
        bytes_fetched=0,
        bytes_evicted=0,
        initial_fetch_transfers=0,
        initial_bytes_fetched=0,
        events=(),
        simulator_comparison=FrozenMapping({}),
    )
    runtime = RuntimeTrace(
        wait_events_inserted=0,
        allocation_requests=0,
        zero_byte_allocation_requests=0,
        materialized_allocation_requests=0,
        free_requests=0,
        record_stream_callbacks=0,
        event_queries=0,
        queued_actions_after=0,
        pending_retirements_after=0,
        callback_failures_after=0,
        step_id=1,
        begin_timestamp_ns=1,
        end_timestamp_ns=2,
        event_capacity=1,
        allocation_event_capacity=1,
        event_overflow=False,
        allocation_event_overflow=False,
        events=(),
    )
    summary = StepTimingSummary(
        profiled_task_seconds=1.0,
        real_task_event_seconds=1.0,
        task_event_delta_seconds=0.0,
        simulated_inter_task_idle_seconds=0.0,
        real_inter_task_idle_seconds=0.0,
        inter_task_idle_delta_seconds=0.0,
        simulated_inter_task_readiness_wait_seconds=0.0,
        real_inter_task_readiness_wait_seconds=0.0,
        real_inter_task_exposed_overhead_seconds=0.0,
        real_initial_readiness_wait_seconds=0.0,
        real_minimum_frontend_lead_seconds=0.0,
        simulated_selected_span_seconds=1.0,
        real_selected_span_seconds=1.0,
        selected_span_delta_seconds=0.0,
        simulator_makespan_seconds=1.0,
        simulator_terminal_tail_seconds=0.0,
        phase_comparisons=(),
        trace_complete=True,
    )
    return StepDiagnostics(
        timing=timing,
        tasks=FrozenMapping({}),
        allocator=allocator,
        transfers=transfers,
        runtime=runtime,
        simulator_comparison=FrozenMapping({}),
        summary=summary,
    )


def test_step_diagnostics_torch_serialization_preserves_immutable_mappings() -> None:
    buffer = io.BytesIO()
    torch.save(_diagnostics(), buffer)
    buffer.seek(0)

    restored = torch.load(buffer, weights_only=False)

    assert isinstance(restored, StepDiagnostics)
    assert isinstance(restored.tasks, FrozenMapping)
    assert isinstance(restored.timing.tasks, FrozenMapping)
    assert isinstance(restored.transfers.simulator_comparison, FrozenMapping)
    assert isinstance(restored.simulator_comparison, FrozenMapping)
    assert restored.as_dict()["schema"] == "shadowspill.step_diagnostics/v4"
