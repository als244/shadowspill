"""Step diagnostics survive torch serialization and export their schema."""

from __future__ import annotations

import io

import torch

from shadowspill.planner.diagnostics.mapping import FrozenMapping
from shadowspill.pytorch.diagnostics.execution import (
    AllocatorTrace,
    LaneSummary,
    RuntimeTrace,
    StepDiagnostics,
    StepTimingSummary,
    Timelines,
    TransferLane,
    TransferRecords,
)
from shadowspill.schema import artifact_schema


def _lane(direction: str) -> TransferLane:
    return TransferLane(
        order=(),
        summary=LaneSummary(
            direction=direction,
            transfers=0,
            bytes=0,
            simulated_busy_seconds=0.0,
            measured_transfers=0,
            stream_busy_seconds=0.0,
            effective_bandwidth_bytes_per_second=None,
            largest_start_delta_seconds=None,
            largest_start_delta_transfer_id=None,
            opening_transfers=0,
            opening_bytes=0,
        ),
    )


def _diagnostics() -> StepDiagnostics:
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
        call_seconds=1.1,
        startup_wait_seconds=0.0,
        initial_actions_seconds=0.0,
        trace_setup_seconds=0.0,
        optimizer_span_seconds=0.1,
        phase_comparisons=(),
        trace_complete=True,
    )
    return StepDiagnostics(
        summary=summary,
        tasks=FrozenMapping({}),
        transfers=TransferRecords(fetch=FrozenMapping({}), evict=FrozenMapping({})),
        timelines=Timelines(
            first_task_start_seconds=0.25,
            compute=(),
            fetch=_lane("fetch"),
            evict=_lane("evict"),
        ),
        allocator=allocator,
        runtime=runtime,
    )


def payload_transfers(diagnostics: StepDiagnostics) -> dict[str, object]:
    value = diagnostics.as_dict()["transfers"]
    assert isinstance(value, dict)
    return value


def test_step_diagnostics_torch_serialization_round_trips() -> None:
    buffer = io.BytesIO()
    torch.save(_diagnostics(), buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=False)
    assert isinstance(restored, StepDiagnostics)
    assert restored.timelines.first_task_start_seconds == 0.25
    assert restored.timelines.fetch.summary.direction == "fetch"
    assert isinstance(restored.tasks, FrozenMapping)
    assert isinstance(restored.transfers.fetch, FrozenMapping)
    assert set(payload_transfers(restored)) == {"fetch", "evict"}
    payload = restored.as_dict()
    assert payload["schema"] == artifact_schema("step_diagnostics")
    assert set(payload) == {
        "schema",
        "summary",
        "tasks",
        "transfers",
        "timelines",
        "allocator",
        "runtime",
    }
    timelines = payload["timelines"]
    assert isinstance(timelines, dict)
    assert set(timelines) == {"clocks", "compute", "fetch", "evict"}
