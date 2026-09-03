"""Immutable evidence for one traced step: summary, timelines, allocator, runtime.

Every time here is seconds. Two clocks appear, and every field says which:

- the device timeline, read from timing events measured against the step's
  origin event on the compute stream. Task markers and transfer intervals
  live here, so compute, fetch, and evict share one zero;
- the host clock, `CLOCK_MONOTONIC`, counted from the runtime trace's
  beginning, which the runtime records at the same point the origin event is
  recorded. Boundary entry and exit, dispatch costs, and the worker's
  queueing and completion observations live here.

Simulated times come from the simulator's own clock, shifted so that the
first selected task starts at zero; `Timelines.first_task_start_seconds` is
where that instant sits on the device timeline, and every delta is taken
after that shift, so a delta reads as drift within the step rather than as
the step's opening cost.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from shadowspill.pytorch.runtime_adapter.telemetry import CapturedAllocationEvent
from shadowspill.pytorch.runtime_adapter.trace import RuntimeTraceEvent
from shadowspill.schema import artifact_schema


@dataclass(frozen=True, slots=True)
class TaskRecord:
    """One selected task, simulated beside measured."""

    #: Identity. `execution_task_id` is the chronological join key shared
    #: with `PlanReport.diagnostics.tasks`; `task_id` names the Program task.
    execution_task_id: str
    task_id: str
    execution_ordinal: int
    semantic_name: str
    phase: str
    microbatch: int | None
    #: Simulated, from the shifted simulator clock: when the task's inputs
    #: were ready, when it started, when it ended, and how long it ran, which
    #: is the isolated profile the plan was built from.
    simulated_ready_seconds: float
    simulated_start_seconds: float
    simulated_end_seconds: float
    expected_profile_seconds: float
    #: Device timeline: when the compute stream reached the task's readiness
    #: marker, when its kernels started after the waits, when they finished,
    #: and how long they ran.
    compute_reached_seconds: float
    compute_started_seconds: float
    compute_finished_seconds: float
    compute_duration_seconds: float
    #: The two waits between reaching the task and starting it: inputs still
    #: being fetched, then ranges its allocations reuse still owned by a
    #: transfer. Both leave the stream idle; the boundary owns both.
    input_readiness_wait_seconds: float
    allocation_reuse_wait_seconds: float
    #: Device minus simulated, after alignment. A start delta that grows along
    #: the lane is drift the simulator did not price; a duration delta is the
    #: profile's error for this task.
    start_delta_seconds: float
    end_delta_seconds: float
    duration_delta_seconds: float
    #: Host clock, from the trace's beginning: the frontend's entry and exit
    #: of the task's two boundaries, which bracket everything it does between
    #: this task's kernels and the next's.
    before_task_enter_seconds: float | None
    before_task_exit_seconds: float | None
    after_task_enter_seconds: float | None
    after_task_exit_seconds: float | None
    #: How long after the frontend handed the task off the stream reached it:
    #: positive is the frontend's lead, zero means the stream caught up.
    frontend_lead_seconds: float | None
    #: Host cost of the boundaries: the complete `before_task` wall time, the
    #: callable's own dispatch, and the complete `after_task` wall time.
    dispatch_before_task_seconds: float
    dispatch_invoke_seconds: float
    dispatch_after_task_seconds: float
    #: The work inside `before_task`, each a disjoint part of it.
    dispatch_input_lookup_seconds: float
    dispatch_storage_rebind_seconds: float
    dispatch_argument_assembly_seconds: float
    #: The work inside `after_task`, each a disjoint part of it.
    dispatch_output_flatten_seconds: float
    dispatch_output_classification_seconds: float
    dispatch_output_adoption_seconds: float
    dispatch_output_state_publish_seconds: float
    dispatch_output_publish_seconds: float
    dispatch_dematerialize_seconds: float
    dispatch_cleanup_seconds: float

    def as_dict(self) -> dict[str, object]:
        return {
            "execution_task_id": self.execution_task_id,
            "task_id": self.task_id,
            "execution_ordinal": self.execution_ordinal,
            "semantic_name": self.semantic_name,
            "phase": self.phase,
            "microbatch": self.microbatch,
            "simulated": {
                "ready_seconds": self.simulated_ready_seconds,
                "start_seconds": self.simulated_start_seconds,
                "end_seconds": self.simulated_end_seconds,
                "duration_seconds": self.expected_profile_seconds,
            },
            "stream": {
                "reached_seconds": self.compute_reached_seconds,
                "started_seconds": self.compute_started_seconds,
                "finished_seconds": self.compute_finished_seconds,
                "duration_seconds": self.compute_duration_seconds,
                "input_readiness_wait_seconds": self.input_readiness_wait_seconds,
                "allocation_reuse_wait_seconds": self.allocation_reuse_wait_seconds,
            },
            "delta": {
                "start_seconds": self.start_delta_seconds,
                "end_seconds": self.end_delta_seconds,
                "duration_seconds": self.duration_delta_seconds,
            },
            "host": {
                "before_task_enter_seconds": self.before_task_enter_seconds,
                "before_task_exit_seconds": self.before_task_exit_seconds,
                "after_task_enter_seconds": self.after_task_enter_seconds,
                "after_task_exit_seconds": self.after_task_exit_seconds,
                "frontend_lead_seconds": self.frontend_lead_seconds,
                "dispatch_before_task_seconds": self.dispatch_before_task_seconds,
                "dispatch_invoke_seconds": self.dispatch_invoke_seconds,
                "dispatch_after_task_seconds": self.dispatch_after_task_seconds,
                "dispatch_input_lookup_seconds": self.dispatch_input_lookup_seconds,
                "dispatch_storage_rebind_seconds": (
                    self.dispatch_storage_rebind_seconds
                ),
                "dispatch_argument_assembly_seconds": (
                    self.dispatch_argument_assembly_seconds
                ),
                "dispatch_output_flatten_seconds": (
                    self.dispatch_output_flatten_seconds
                ),
                "dispatch_output_classification_seconds": (
                    self.dispatch_output_classification_seconds
                ),
                "dispatch_output_adoption_seconds": (
                    self.dispatch_output_adoption_seconds
                ),
                "dispatch_output_state_publish_seconds": (
                    self.dispatch_output_state_publish_seconds
                ),
                "dispatch_output_publish_seconds": (
                    self.dispatch_output_publish_seconds
                ),
                "dispatch_dematerialize_seconds": self.dispatch_dematerialize_seconds,
                "dispatch_cleanup_seconds": self.dispatch_cleanup_seconds,
            },
        }


@dataclass(frozen=True, slots=True)
class TransferRecord:
    """One scheduled transfer on its lane, simulated beside measured."""

    #: Identity. A scheduled transfer's `transfer_id` is
    #: `<direction>_<sequence>`, with `sequence` its FIFO position among the
    #: plan's transfers on the lane; an opening transfer's is
    #: `<direction>_opening_<index>`. `triggered_by` is what released the
    #: transfer: the execution task id of the task whose completion did, a
    #: key into `tasks`, or `init` for the opening placement batch the
    #: runtime issues before the first task.
    transfer_id: str
    direction: str
    sequence: int
    triggered_by: str
    alias_group_id: str
    bytes: int
    #: The object's place in the step, by execution task id. `previous_access`
    #: is the last selected task, up to and including the trigger, that read
    #: or wrote the object, and `next_access` the first later task that does;
    #: `modified_by` is the last one, up to the trigger, that created or
    #: mutated it. `init` stands for no such task before the transfer, so the
    #: bytes are what the step was given; `persistent` for none after it
    #: within this call, so the object outlives the step. A fetch exists for
    #: its next access; an eviction saves what its modifier produced.
    previous_access: str
    next_access: str
    modified_by: str
    #: Simulated, from the shifted simulator clock: when the transfer could
    #: start, when the lane started it, when it ended, and its priced
    #: duration at the assumed lane bandwidth. `None` for an opening
    #: transfer, which the simulator does not model.
    simulated_ready_seconds: float | None
    simulated_start_seconds: float | None
    simulated_end_seconds: float | None
    simulated_duration_seconds: float | None
    #: Device timeline: the copy's interval on the lane, bracketed by timing
    #: events the worker recorded around it. `None` when the trace could not
    #: measure this transfer.
    stream_start_seconds: float | None
    stream_end_seconds: float | None
    stream_duration_seconds: float | None
    #: Device minus simulated, after alignment; `None` without a stream
    #: interval or without a simulation.
    start_delta_seconds: float | None
    end_delta_seconds: float | None
    duration_delta_seconds: float | None
    #: Host clock, from the trace's beginning: when the action was queued,
    #: when its destination was reserved, when the worker dispatched the copy
    #: to the lane, and when the worker observed its completion.
    queued_seconds: float | None
    reserved_seconds: float | None
    dispatched_seconds: float
    completion_observed_seconds: float

    def as_dict(self) -> dict[str, object]:
        return {
            "transfer_id": self.transfer_id,
            "direction": self.direction,
            "sequence": self.sequence,
            "triggered_by": self.triggered_by,
            "alias_group_id": self.alias_group_id,
            "bytes": self.bytes,
            "previous_access": self.previous_access,
            "next_access": self.next_access,
            "modified_by": self.modified_by,
            "simulated": {
                "ready_seconds": self.simulated_ready_seconds,
                "start_seconds": self.simulated_start_seconds,
                "end_seconds": self.simulated_end_seconds,
                "duration_seconds": self.simulated_duration_seconds,
            },
            "stream": {
                "start_seconds": self.stream_start_seconds,
                "end_seconds": self.stream_end_seconds,
                "duration_seconds": self.stream_duration_seconds,
            },
            "delta": {
                "start_seconds": self.start_delta_seconds,
                "end_seconds": self.end_delta_seconds,
                "duration_seconds": self.duration_delta_seconds,
            },
            "host": {
                "queued_seconds": self.queued_seconds,
                "reserved_seconds": self.reserved_seconds,
                "dispatched_seconds": self.dispatched_seconds,
                "completion_observed_seconds": self.completion_observed_seconds,
            },
        }


@dataclass(frozen=True, slots=True)
class LaneSummary:
    """What one transfer lane did over the step, simulated beside measured."""

    direction: str
    transfers: int
    bytes: int
    #: Lane time the simulator priced for the scheduled transfers.
    simulated_busy_seconds: float
    #: How many transfers carry a stream interval, and the lane time those
    #: intervals add up to. The effective bandwidth is their bytes over that
    #: time; compare it with the bandwidth the plan assumed, which the plan
    #: summary states.
    measured_transfers: int
    stream_busy_seconds: float
    effective_bandwidth_bytes_per_second: float | None
    #: The largest start delta on the lane, signed, and the transfer that
    #: reached it: where the lane had drifted furthest from the simulation.
    largest_start_delta_seconds: float | None
    largest_start_delta_transfer_id: str | None
    #: The opening placement batch on this lane: transfers the runtime issued
    #: before the first task to restore the step's initial objects, which
    #: precede the span and carry no simulation. They are records too,
    #: `triggered_by` `init`, and lead the lane's order.
    opening_transfers: int
    opening_bytes: int

    def as_dict(self) -> dict[str, object]:
        return {
            "direction": self.direction,
            "transfers": self.transfers,
            "bytes": self.bytes,
            "simulated_busy_seconds": self.simulated_busy_seconds,
            "measured_transfers": self.measured_transfers,
            "stream_busy_seconds": self.stream_busy_seconds,
            "effective_bandwidth_bytes_per_second": (
                self.effective_bandwidth_bytes_per_second
            ),
            "largest_start_delta_seconds": self.largest_start_delta_seconds,
            "largest_start_delta_transfer_id": self.largest_start_delta_transfer_id,
            "opening_transfers": self.opening_transfers,
            "opening_bytes": self.opening_bytes,
        }


@dataclass(frozen=True, slots=True)
class TransferRecords:
    """Every scheduled transfer, by transfer id, grouped by direction."""

    fetch: Mapping[str, TransferRecord]
    evict: Mapping[str, TransferRecord]

    def as_dict(self) -> dict[str, object]:
        return {
            "fetch": {key: item.as_dict() for key, item in self.fetch.items()},
            "evict": {key: item.as_dict() for key, item in self.evict.items()},
        }


@dataclass(frozen=True, slots=True)
class TransferLane:
    """One direction's transfers in FIFO order, with the lane's summary.

    `order` holds transfer ids into the lane's group of
    `StepDiagnostics.transfers`; an id's position is the transfer's
    `sequence` on the lane.
    """

    order: tuple[str, ...]
    summary: LaneSummary

    def as_dict(self) -> dict[str, object]:
        return {"summary": self.summary.as_dict(), "order": list(self.order)}


@dataclass(frozen=True, slots=True)
class Timelines:
    """The step on three lanes, each in its stream's order, sharing one zero.

    The lanes hold references: `compute` is every selected task's execution
    task id in compute-stream order, keys into `StepDiagnostics.tasks`;
    `fetch` and `evict` list transfer ids in each lane's FIFO order, keys
    into the same-named group of `StepDiagnostics.transfers`. Device times
    in those records count from the origin event; simulated times count from
    the first selected task's simulated start. `first_task_start_seconds` is
    when that task's kernels actually started on the device: the step's
    prologue, which every invocation pays and the simulator does not model --
    the opening restore of the first task's inputs, input staging, and the
    first dispatch. Every delta is taken after shifting the simulation to it,
    so a delta is drift within the step and the prologue is read here, once.
    """

    first_task_start_seconds: float
    compute: tuple[str, ...]
    fetch: TransferLane
    evict: TransferLane

    def as_dict(self) -> dict[str, object]:
        return {
            "clocks": {
                "stream": "seconds from the step origin event on the device",
                "simulated": ("seconds from the first selected task's simulated start"),
                "host": "seconds from the runtime trace's beginning",
                "first_task_start_seconds": self.first_task_start_seconds,
            },
            "compute": list(self.compute),
            "fetch": self.fetch.as_dict(),
            "evict": self.evict.as_dict(),
        }


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
class RuntimeTrace:
    """Runtime counter changes, terminal queue state, and the raw trace."""

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
    #: The step as the caller saw it, on the host clock: the whole planned
    #: call, the wait for the previous invocation to drain at its start, the
    #: submission of the opening placement batch, and the trace's one-time
    #: setup, which only the first traced call pays.
    call_seconds: float
    startup_wait_seconds: float
    initial_actions_seconds: float
    trace_setup_seconds: float
    #: First optimizer task's start through the last one's end, on the
    #: device timeline.
    optimizer_span_seconds: float
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
            "call_seconds": self.call_seconds,
            "startup_wait_seconds": self.startup_wait_seconds,
            "initial_actions_seconds": self.initial_actions_seconds,
            "trace_setup_seconds": self.trace_setup_seconds,
            "optimizer_span_seconds": self.optimizer_span_seconds,
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

    summary: StepTimingSummary
    #: Every selected task by execution task id, and every scheduled transfer
    #: by transfer id grouped by direction; the timelines refer into both.
    tasks: Mapping[str, TaskRecord]
    transfers: TransferRecords
    timelines: Timelines
    allocator: AllocatorTrace
    runtime: RuntimeTrace

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": artifact_schema("step_diagnostics"),
            "summary": self.summary.as_dict(),
            "tasks": {key: item.as_dict() for key, item in self.tasks.items()},
            "transfers": self.transfers.as_dict(),
            "timelines": self.timelines.as_dict(),
            "allocator": self.allocator.as_dict(),
            "runtime": self.runtime.as_dict(),
        }


__all__ = [
    "AllocatorTrace",
    "LaneSummary",
    "PhaseTimingComparison",
    "RuntimeTrace",
    "StepDiagnostics",
    "StepTimingSummary",
    "TaskRecord",
    "Timelines",
    "TransferLane",
    "TransferRecord",
    "TransferRecords",
]
