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
    task_compute_seconds: float | None
    before_readiness_waits_sequence: int
    before_task_compute_sequence: int
    after_task_compute_sequence: int
    native_before_task_enter_seconds: float | None
    native_before_task_exit_seconds: float | None
    native_after_task_enter_seconds: float | None
    native_after_task_exit_seconds: float | None
    host_before_task_seconds: float
    host_stream_resolution_seconds: float
    host_readiness_marker_seconds: float
    host_native_before_task_seconds: float
    host_input_lookup_seconds: float
    host_storage_rebind_seconds: float
    host_generation_publish_seconds: float
    host_argument_assembly_seconds: float
    host_rebind_seconds: float
    host_dispatch_seconds: float
    host_output_flatten_seconds: float
    host_output_classification_seconds: float
    host_output_adoption_seconds: float
    host_output_state_publish_seconds: float
    host_gradient_accumulation_seconds: float
    host_output_publish_seconds: float
    host_dematerialize_seconds: float
    host_postprocess_seconds: float
    host_native_after_task_seconds: float
    host_cleanup_seconds: float
    host_after_task_seconds: float
    host_total_seconds: float

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
            "task_compute_seconds": self.task_compute_seconds,
            "before_readiness_waits_sequence": (self.before_readiness_waits_sequence),
            "before_task_compute_sequence": self.before_task_compute_sequence,
            "after_task_compute_sequence": self.after_task_compute_sequence,
            "native_before_task_enter_seconds": (self.native_before_task_enter_seconds),
            "native_before_task_exit_seconds": self.native_before_task_exit_seconds,
            "native_after_task_enter_seconds": self.native_after_task_enter_seconds,
            "native_after_task_exit_seconds": self.native_after_task_exit_seconds,
            "host_before_task_seconds": self.host_before_task_seconds,
            "host_stream_resolution_seconds": self.host_stream_resolution_seconds,
            "host_readiness_marker_seconds": self.host_readiness_marker_seconds,
            "host_native_before_task_seconds": (self.host_native_before_task_seconds),
            "host_input_lookup_seconds": self.host_input_lookup_seconds,
            "host_storage_rebind_seconds": self.host_storage_rebind_seconds,
            "host_generation_publish_seconds": (self.host_generation_publish_seconds),
            "host_argument_assembly_seconds": self.host_argument_assembly_seconds,
            "host_rebind_seconds": self.host_rebind_seconds,
            "host_dispatch_seconds": self.host_dispatch_seconds,
            "host_output_flatten_seconds": self.host_output_flatten_seconds,
            "host_output_classification_seconds": (
                self.host_output_classification_seconds
            ),
            "host_output_adoption_seconds": self.host_output_adoption_seconds,
            "host_output_state_publish_seconds": (
                self.host_output_state_publish_seconds
            ),
            "host_gradient_accumulation_seconds": (
                self.host_gradient_accumulation_seconds
            ),
            "host_output_publish_seconds": self.host_output_publish_seconds,
            "host_dematerialize_seconds": self.host_dematerialize_seconds,
            "host_postprocess_seconds": self.host_postprocess_seconds,
            "host_native_after_task_seconds": self.host_native_after_task_seconds,
            "host_cleanup_seconds": self.host_cleanup_seconds,
            "host_after_task_seconds": self.host_after_task_seconds,
            "host_total_seconds": self.host_total_seconds,
        }


@dataclass(frozen=True, slots=True)
class ExecutionTiming:
    """Qualification-only decomposition of one accumulated training call."""

    compute_seconds: float
    optimizer_seconds: float
    host_call_seconds: float
    host_startup_wait_seconds: float
    host_initial_actions_seconds: float
    trace_setup_seconds: float
    phase_gpu_seconds: tuple[tuple[str, float], ...]
    tasks: Mapping[str, TaskExecutionTiming]

    def as_dict(self, *, include_tasks: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "compute_seconds": self.compute_seconds,
            "optimizer_seconds": self.optimizer_seconds,
            "host_call_seconds": self.host_call_seconds,
            "host_startup_wait_seconds": self.host_startup_wait_seconds,
            "host_initial_actions_seconds": self.host_initial_actions_seconds,
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
    events: tuple[RuntimeTraceEvent, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "actions": [item.to_dict() for item in self.actions],
            "fetch_transfers": self.fetch_transfers,
            "evict_transfers": self.evict_transfers,
            "bytes_fetched": self.bytes_fetched,
            "bytes_evicted": self.bytes_evicted,
            "events": [item.as_dict() for item in self.events],
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
    expected_profile_seconds: float
    observed_gpu_seconds: float
    delta_seconds: float

    def as_dict(self) -> dict[str, object]:
        return {
            "execution_task_id": self.execution_task_id,
            "task_id": self.task_id,
            "expected_profile_seconds": self.expected_profile_seconds,
            "observed_gpu_seconds": self.observed_gpu_seconds,
            "delta_seconds": self.delta_seconds,
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

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "shadowspill.step_diagnostics/v1",
            "timing": self.timing.as_dict(include_tasks=False),
            "tasks": {
                execution_task_id: item.as_dict()
                for execution_task_id, item in self.tasks.items()
            },
            "allocator": self.allocator.as_dict(),
            "transfers": self.transfers.as_dict(),
            "runtime": self.runtime.as_dict(),
            "simulator_comparison": {
                execution_task_id: item.as_dict()
                for execution_task_id, item in self.simulator_comparison.items()
            },
        }


__all__ = [
    "AllocatorTrace",
    "ExecutionTiming",
    "RuntimeTrace",
    "SimulatorTaskComparison",
    "StepDiagnostics",
    "TaskExecutionTiming",
    "TransferTrace",
]
