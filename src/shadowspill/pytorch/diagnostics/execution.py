"""Immutable timing, allocator, transfer, and runtime evidence for one step."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from shadowspill.ir import MemoryAction
from shadowspill.pytorch.runtime_adapter.telemetry import CapturedAllocationEvent
from shadowspill.pytorch.runtime_adapter.trace import RuntimeTraceEvent


@dataclass(frozen=True, slots=True)
class TaskExecutionTiming:
    """Qualification timing for one selected task invocation."""

    task_id: str
    execution_ordinal: int
    execution_task_id: str
    semantic_name: str
    phase: str
    microbatch: int | None
    expected_profile_seconds: float
    before_task_enter_timestamp_ns: int
    before_task_exit_timestamp_ns: int
    after_task_enter_timestamp_ns: int
    after_task_exit_timestamp_ns: int
    before_readiness_waits_timestamp_ns: int
    before_task_compute_timestamp_ns: int
    after_task_compute_timestamp_ns: int
    gpu_start_seconds: float
    gpu_end_seconds: float
    gpu_duration_seconds: float
    before_readiness_waits_seconds: float | None
    before_task_compute_seconds: float | None
    after_task_compute_seconds: float | None
    readiness_wait_seconds: float | None
    before_readiness_waits_sequence: int
    before_task_compute_sequence: int
    after_task_compute_sequence: int
    runtime_before_task_enter_seconds: float | None
    runtime_before_task_exit_seconds: float | None
    runtime_after_task_enter_seconds: float | None
    runtime_after_task_exit_seconds: float | None
    #: How long after the frontend handed this task off the compute stream
    #: arrived at it. Positive means the frontend was that far ahead; at zero
    #: the stream has caught up and the next thing it waits on is the frontend
    #: itself. Measured to the stream reaching the task, so a task that then
    #: waits for its inputs still counts as having been handed off in time.
    frontend_lead_seconds: float | None
    dispatch_before_task_seconds: float
    dispatch_stream_resolution_seconds: float
    dispatch_readiness_marker_seconds: float
    dispatch_runtime_before_task_seconds: float
    dispatch_input_lookup_seconds: float
    dispatch_storage_rebind_seconds: float
    dispatch_argument_assembly_seconds: float
    dispatch_rebind_seconds: float
    dispatch_invoke_seconds: float
    dispatch_output_flatten_seconds: float
    dispatch_output_classification_seconds: float
    dispatch_output_adoption_seconds: float
    dispatch_output_state_publish_seconds: float
    dispatch_gradient_accumulation_seconds: float
    dispatch_output_publish_seconds: float
    dispatch_dematerialize_seconds: float
    dispatch_postprocess_seconds: float
    dispatch_runtime_after_task_seconds: float
    dispatch_cleanup_seconds: float
    dispatch_after_task_seconds: float
    dispatch_total_seconds: float

    def as_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "execution_ordinal": self.execution_ordinal,
            "execution_task_id": self.execution_task_id,
            "semantic_name": self.semantic_name,
            "phase": self.phase,
            "microbatch": self.microbatch,
            "expected_profile_seconds": self.expected_profile_seconds,
            "boundary_timestamps": {
                "unit": "nanoseconds",
                "host": {
                    "clock": "CLOCK_MONOTONIC",
                    "before_task": {
                        "enter": self.before_task_enter_timestamp_ns,
                        "exit": self.before_task_exit_timestamp_ns,
                    },
                    "after_task": {
                        "enter": self.after_task_enter_timestamp_ns,
                        "exit": self.after_task_exit_timestamp_ns,
                    },
                },
                "compute_stream": {
                    "clock": "cuda_event_elapsed_from_step_origin",
                    "before_readiness_waits": (
                        self.before_readiness_waits_timestamp_ns
                    ),
                    "before_task_compute": self.before_task_compute_timestamp_ns,
                    "after_task_compute": self.after_task_compute_timestamp_ns,
                },
            },
            "gpu_start_seconds": self.gpu_start_seconds,
            "gpu_end_seconds": self.gpu_end_seconds,
            "gpu_duration_seconds": self.gpu_duration_seconds,
            "before_readiness_waits_seconds": self.before_readiness_waits_seconds,
            "before_task_compute_seconds": self.before_task_compute_seconds,
            "after_task_compute_seconds": self.after_task_compute_seconds,
            "readiness_wait_seconds": self.readiness_wait_seconds,
            "before_readiness_waits_sequence": (self.before_readiness_waits_sequence),
            "before_task_compute_sequence": self.before_task_compute_sequence,
            "after_task_compute_sequence": self.after_task_compute_sequence,
            "runtime_before_task_enter_seconds": (
                self.runtime_before_task_enter_seconds
            ),
            "runtime_before_task_exit_seconds": self.runtime_before_task_exit_seconds,
            "runtime_after_task_enter_seconds": self.runtime_after_task_enter_seconds,
            "runtime_after_task_exit_seconds": self.runtime_after_task_exit_seconds,
            "frontend_lead_seconds": self.frontend_lead_seconds,
            "dispatch_before_task_seconds": self.dispatch_before_task_seconds,
            "dispatch_stream_resolution_seconds": (
                self.dispatch_stream_resolution_seconds
            ),
            "dispatch_readiness_marker_seconds": self.dispatch_readiness_marker_seconds,
            "dispatch_runtime_before_task_seconds": (
                self.dispatch_runtime_before_task_seconds
            ),
            "dispatch_input_lookup_seconds": self.dispatch_input_lookup_seconds,
            "dispatch_storage_rebind_seconds": self.dispatch_storage_rebind_seconds,
            "dispatch_argument_assembly_seconds": (
                self.dispatch_argument_assembly_seconds
            ),
            "dispatch_rebind_seconds": self.dispatch_rebind_seconds,
            "dispatch_invoke_seconds": self.dispatch_invoke_seconds,
            "dispatch_output_flatten_seconds": self.dispatch_output_flatten_seconds,
            "dispatch_output_classification_seconds": (
                self.dispatch_output_classification_seconds
            ),
            "dispatch_output_adoption_seconds": self.dispatch_output_adoption_seconds,
            "dispatch_output_state_publish_seconds": (
                self.dispatch_output_state_publish_seconds
            ),
            "dispatch_gradient_accumulation_seconds": (
                self.dispatch_gradient_accumulation_seconds
            ),
            "dispatch_output_publish_seconds": self.dispatch_output_publish_seconds,
            "dispatch_dematerialize_seconds": self.dispatch_dematerialize_seconds,
            "dispatch_postprocess_seconds": self.dispatch_postprocess_seconds,
            "dispatch_runtime_after_task_seconds": (
                self.dispatch_runtime_after_task_seconds
            ),
            "dispatch_cleanup_seconds": self.dispatch_cleanup_seconds,
            "dispatch_after_task_seconds": self.dispatch_after_task_seconds,
            "dispatch_total_seconds": self.dispatch_total_seconds,
        }


@dataclass(frozen=True, slots=True)
class ExecutionTiming:
    """Qualification-only decomposition of one accumulated training call."""

    compute_seconds: float
    optimizer_seconds: float
    dispatch_call_seconds: float
    dispatch_startup_wait_seconds: float
    dispatch_initial_actions_seconds: float
    trace_setup_seconds: float
    phase_gpu_seconds: tuple[tuple[str, float], ...]
    tasks: Mapping[str, TaskExecutionTiming]

    def as_dict(self, *, include_tasks: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "compute_seconds": self.compute_seconds,
            "optimizer_seconds": self.optimizer_seconds,
            "dispatch_call_seconds": self.dispatch_call_seconds,
            "dispatch_startup_wait_seconds": self.dispatch_startup_wait_seconds,
            "dispatch_initial_actions_seconds": self.dispatch_initial_actions_seconds,
            "trace_setup_seconds": self.trace_setup_seconds,
            "phase_gpu_seconds": dict(self.phase_gpu_seconds),
        }
        if include_tasks:
            result["tasks"] = {
                execution_task_id: item.as_dict()
                for execution_task_id, item in self.tasks.items()
            }
        return result


@dataclass(frozen=True, slots=True)
class AllocatorTrace:
    """Ordered allocator lifetimes and before/after slab state."""

    events: tuple[CapturedAllocationEvent, ...]
    live_allocations_before: int
    live_allocations_after: int
    allocated_bytes_before: int
    allocated_bytes_after: int
    peak_allocated_bytes: int
    free_bytes_after: int
    free_prefix_bytes_after: int
    largest_free_range_bytes_after: int
    external_fragmentation_bytes_after: int
    blocked_allocators_after: int
    overflow: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "events": [
                {
                    "sequence": item.sequence,
                    "task_id": item.task_id,
                    "allocation_id": item.allocation_id,
                    "generation": item.generation,
                    "requested_bytes": item.requested_bytes,
                    "charged_bytes": item.charged_bytes,
                    "slab_offset": item.slab_offset,
                    "kind": item.kind.name.lower(),
                    "category": item.category.name.lower(),
                }
                for item in self.events
            ],
            "live_allocations_before": self.live_allocations_before,
            "live_allocations_after": self.live_allocations_after,
            "allocated_bytes_before": self.allocated_bytes_before,
            "allocated_bytes_after": self.allocated_bytes_after,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "free_bytes_after": self.free_bytes_after,
            "free_prefix_bytes_after": self.free_prefix_bytes_after,
            "largest_free_range_bytes_after": self.largest_free_range_bytes_after,
            "external_fragmentation_bytes_after": (
                self.external_fragmentation_bytes_after
            ),
            "blocked_allocators_after": self.blocked_allocators_after,
            "overflow": self.overflow,
        }


@dataclass(frozen=True, slots=True)
class TransferTrace:
    """Annotated actions and completed-transfer counter deltas."""

    actions: tuple[MemoryAction, ...]
    fetch_transfers: int
    evict_transfers: int
    bytes_fetched: int
    bytes_evicted: int
    initial_fetch_transfers: int
    initial_bytes_fetched: int
    events: tuple[RuntimeTraceEvent, ...]
    simulator_comparison: Mapping[str, SimulatorTransferComparison]

    def as_dict(self) -> dict[str, object]:
        return {
            "actions": [item.to_dict() for item in self.actions],
            "fetch_transfers": self.fetch_transfers,
            "evict_transfers": self.evict_transfers,
            "bytes_fetched": self.bytes_fetched,
            "bytes_evicted": self.bytes_evicted,
            "initial_fetch_transfers": self.initial_fetch_transfers,
            "initial_bytes_fetched": self.initial_bytes_fetched,
            "events": [item.as_dict() for item in self.events],
            "simulator_comparison": {
                transfer_id: item.as_dict()
                for transfer_id, item in self.simulator_comparison.items()
            },
        }


@dataclass(frozen=True, slots=True)
class RuntimeTrace:
    """Runtime counter changes and terminal queue state for the traced call."""

    wait_events_inserted: int
    allocation_requests: int
    zero_byte_allocation_requests: int
    materialized_allocation_requests: int
    free_requests: int
    record_stream_callbacks: int
    event_queries: int
    queued_actions_after: int
    pending_retirements_after: int
    callback_failures_after: int
    step_id: int
    begin_timestamp_ns: int
    end_timestamp_ns: int
    event_capacity: int
    allocation_event_capacity: int
    event_overflow: bool
    allocation_event_overflow: bool
    events: tuple[RuntimeTraceEvent, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "wait_events_inserted": self.wait_events_inserted,
            "allocation_requests": self.allocation_requests,
            "zero_byte_allocation_requests": self.zero_byte_allocation_requests,
            "materialized_allocation_requests": self.materialized_allocation_requests,
            "free_requests": self.free_requests,
            "record_stream_callbacks": self.record_stream_callbacks,
            "event_queries": self.event_queries,
            "queued_actions_after": self.queued_actions_after,
            "pending_retirements_after": self.pending_retirements_after,
            "callback_failures_after": self.callback_failures_after,
            "step_id": self.step_id,
            "begin_timestamp_ns": self.begin_timestamp_ns,
            "end_timestamp_ns": self.end_timestamp_ns,
            "event_capacity": self.event_capacity,
            "allocation_event_capacity": self.allocation_event_capacity,
            "event_overflow": self.event_overflow,
            "allocation_event_overflow": self.allocation_event_overflow,
            "events": [item.as_dict() for item in self.events],
        }


@dataclass(frozen=True, slots=True)
class SimulatorTaskComparison:
    execution_task_id: str
    task_id: str
    simulated_start_ns: int
    simulated_end_ns: int
    simulated_start_seconds: float
    real_start_seconds: float
    start_delta_seconds: float
    simulated_end_seconds: float
    real_end_seconds: float
    end_delta_seconds: float
    expected_profile_seconds: float
    observed_gpu_seconds: float
    duration_delta_seconds: float

    def as_dict(self) -> dict[str, object]:
        return {
            "execution_task_id": self.execution_task_id,
            "task_id": self.task_id,
            "simulated_start_ns": self.simulated_start_ns,
            "simulated_end_ns": self.simulated_end_ns,
            "simulated_start_seconds": self.simulated_start_seconds,
            "real_start_seconds": self.real_start_seconds,
            "start_delta_seconds": self.start_delta_seconds,
            "simulated_end_seconds": self.simulated_end_seconds,
            "real_end_seconds": self.real_end_seconds,
            "end_delta_seconds": self.end_delta_seconds,
            "expected_profile_seconds": self.expected_profile_seconds,
            "observed_gpu_seconds": self.observed_gpu_seconds,
            "duration_delta_seconds": self.duration_delta_seconds,
        }


@dataclass(frozen=True, slots=True)
class SimulatorTransferComparison:
    """Simulator interval versus the worker-observed real transfer frontier."""

    transfer_id: str
    direction: str
    sequence: int
    trigger_task_id: str
    execution_task_id: str
    alias_group_id: str
    bytes: int
    simulated_ready_seconds: float
    simulated_start_seconds: float
    simulated_end_seconds: float
    simulated_duration_seconds: float
    real_queued_seconds: float | None
    real_reserved_seconds: float | None
    real_dispatch_timestamp_ns: int
    real_completion_timestamp_ns: int
    real_dispatch_seconds: float
    real_completion_seconds: float
    real_frontier_duration_seconds: float
    start_delta_seconds: float
    end_delta_seconds: float
    duration_delta_seconds: float

    def as_dict(self) -> dict[str, object]:
        return {
            "transfer_id": self.transfer_id,
            "direction": self.direction,
            "sequence": self.sequence,
            "trigger_task_id": self.trigger_task_id,
            "execution_task_id": self.execution_task_id,
            "alias_group_id": self.alias_group_id,
            "bytes": self.bytes,
            "timing_basis": (
                "simulator_lane_interval_vs_dispatch_worker_frontier; "
                "aligned_to_first_scheduled_transfer"
            ),
            "simulated_ready_seconds": self.simulated_ready_seconds,
            "simulated_start_seconds": self.simulated_start_seconds,
            "simulated_end_seconds": self.simulated_end_seconds,
            "simulated_duration_seconds": self.simulated_duration_seconds,
            "real_queued_seconds": self.real_queued_seconds,
            "real_reserved_seconds": self.real_reserved_seconds,
            "real_dispatch_timestamp_ns": self.real_dispatch_timestamp_ns,
            "real_completion_timestamp_ns": self.real_completion_timestamp_ns,
            "real_dispatch_seconds": self.real_dispatch_seconds,
            "real_completion_seconds": self.real_completion_seconds,
            "real_frontier_duration_seconds": self.real_frontier_duration_seconds,
            "start_delta_seconds": self.start_delta_seconds,
            "end_delta_seconds": self.end_delta_seconds,
            "duration_delta_seconds": self.duration_delta_seconds,
        }


@dataclass(frozen=True, slots=True)
class PhaseTimingComparison:
    """Expected and observed task-event time for one semantic phase."""

    phase: str
    profiled_task_seconds: float
    real_task_event_seconds: float
    delta_seconds: float

    def as_dict(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "profiled_task_seconds": self.profiled_task_seconds,
            "real_task_event_seconds": self.real_task_event_seconds,
            "delta_seconds": self.delta_seconds,
        }


@dataclass(frozen=True, slots=True)
class StepTimingSummary:
    """Compact simulator-versus-runtime reconciliation for one traced step."""

    profiled_task_seconds: float
    real_task_event_seconds: float
    task_event_delta_seconds: float
    #: Compute-stream idle inside the span, and the two things that cause it.
    #: A task waits for its inputs to be resident, or the stream has not
    #: reached the task yet. The two account for the idle exactly.
    simulated_inter_task_idle_seconds: float
    real_inter_task_idle_seconds: float
    inter_task_idle_delta_seconds: float
    simulated_inter_task_readiness_wait_seconds: float
    real_inter_task_readiness_wait_seconds: float
    #: The rest of the interval: the frontend had not reached the next task,
    #: so nothing was on the stream to run. Exposed, because the frontend does
    #: this work at every boundary and running ahead hides whatever its lead
    #: covers; what is left here is the shortfall, not the cost of the work.
    #: This says nothing about a stream left idle inside a task, which no field
    #: here measures. Where the frontend stayed ahead it is the floor between
    #: two stream event records rather than a cost -- measured at a quarter of
    #: a microsecond to about one -- so read a microsecond or two as nothing
    #: and anything above that as real.
    real_inter_task_exposed_overhead_seconds: float
    #: The first task's wait ends where the span begins, so this is the one
    #: cost the span cannot contain: the fetches a step opens with.
    real_initial_readiness_wait_seconds: float
    #: The smallest lead any task had. The exposed overhead above is what
    #: happens once this reaches zero, so it is the margin that was left.
    real_minimum_frontend_lead_seconds: float
    simulated_selected_span_seconds: float
    real_selected_span_seconds: float
    selected_span_delta_seconds: float
    simulator_makespan_seconds: float
    simulator_terminal_tail_seconds: float
    phase_comparisons: tuple[PhaseTimingComparison, ...]
    trace_complete: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "profiled_task_seconds": self.profiled_task_seconds,
            "real_task_event_seconds": self.real_task_event_seconds,
            "task_event_delta_seconds": self.task_event_delta_seconds,
            "simulated_inter_task_idle_seconds": (
                self.simulated_inter_task_idle_seconds
            ),
            "real_inter_task_idle_seconds": self.real_inter_task_idle_seconds,
            "inter_task_idle_delta_seconds": self.inter_task_idle_delta_seconds,
            "simulated_inter_task_readiness_wait_seconds": (
                self.simulated_inter_task_readiness_wait_seconds
            ),
            "real_inter_task_readiness_wait_seconds": (
                self.real_inter_task_readiness_wait_seconds
            ),
            "real_inter_task_exposed_overhead_seconds": (
                self.real_inter_task_exposed_overhead_seconds
            ),
            "real_initial_readiness_wait_seconds": (
                self.real_initial_readiness_wait_seconds
            ),
            "real_minimum_frontend_lead_seconds": (
                self.real_minimum_frontend_lead_seconds
            ),
            "simulated_selected_span_seconds": self.simulated_selected_span_seconds,
            "real_selected_span_seconds": self.real_selected_span_seconds,
            "selected_span_delta_seconds": self.selected_span_delta_seconds,
            "simulator_makespan_seconds": self.simulator_makespan_seconds,
            "simulator_terminal_tail_seconds": self.simulator_terminal_tail_seconds,
            "phase_comparisons": {
                item.phase: item.as_dict() for item in self.phase_comparisons
            },
            "trace_complete": self.trace_complete,
        }


@dataclass(frozen=True, slots=True)
class StepDiagnostics:
    """Resolved, immutable detailed evidence for one real training call.

    The unresolved :class:`DiagnosticsHandle` is asynchronous. Resolving it is
    explicitly synchronizing because callback and CUDA-event records must have
    completed before they can be copied safely.
    """

    timing: ExecutionTiming
    tasks: Mapping[str, TaskExecutionTiming]
    allocator: AllocatorTrace
    transfers: TransferTrace
    runtime: RuntimeTrace
    simulator_comparison: Mapping[str, SimulatorTaskComparison]
    summary: StepTimingSummary

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "shadowspill.step_diagnostics/v2",
            "timing": self.timing.as_dict(include_tasks=False),
            "tasks": {
                execution_task_id: item.as_dict()
                for execution_task_id, item in self.tasks.items()
            },
            "allocator": self.allocator.as_dict(),
            "transfers": self.transfers.as_dict(),
            "runtime": self.runtime.as_dict(),
            "summary": self.summary.as_dict(),
            "simulator_comparison": {
                execution_task_id: item.as_dict()
                for execution_task_id, item in self.simulator_comparison.items()
            },
        }


__all__ = [
    "AllocatorTrace",
    "ExecutionTiming",
    "PhaseTimingComparison",
    "RuntimeTrace",
    "SimulatorTaskComparison",
    "SimulatorTransferComparison",
    "StepDiagnostics",
    "StepTimingSummary",
    "TaskExecutionTiming",
    "TransferTrace",
]
