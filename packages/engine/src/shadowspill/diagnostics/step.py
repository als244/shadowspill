"""Immutable evidence for one traced step: summary, timelines, allocator, runtime.

Times here are seconds; the collective ledger retains integer nanoseconds.
Two clocks appear, and every field says which:

- the device timeline, read from timing events measured against the step's
  origin event on the compute stream. Task markers and transfer intervals
  live here, so compute, fetch, and evict share one zero;
- the host clock, `CLOCK_MONOTONIC`, counted from a timestamp immediately
  before the invocation origin is recorded. Boundary and worker observations
  live here.

Simulated times start at invocation entry too. Deltas subtract those unshifted
coordinates, so entry delay and accumulated dispatch drift stay visible.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from shadowspill.runtime.abi.statistics import LaneStatistics
from shadowspill.runtime.telemetry import CapturedAllocationEvent
from shadowspill.runtime.trace import RuntimeTraceEvent
from shadowspill.schema import artifact_schema

from .collectives import CollectiveTrace


@dataclass(frozen=True, slots=True)
class TaskRecord:
    """One selected task, simulated beside measured."""

    #: `execution_task_id` identifies the chronological position in this run;
    #: `task_id` joins the record to the selected task in the saved Plan.
    execution_task_id: str
    task_id: str
    execution_ordinal: int
    semantic_name: str
    phase: str
    microbatch: int | None
    #: Simulated, from invocation entry: when the task's inputs
    #: were ready, when it started, when it ended, and how long it ran, which
    #: is the isolated profile the plan was built from.
    simulated_ready_at_seconds: float
    simulated_started_at_seconds: float
    simulated_finished_at_seconds: float
    expected_profile_seconds: float
    #: Device timeline: when the compute stream reached the task's readiness
    #: marker, when its kernels started after the waits, when they finished,
    #: and how long they ran.
    compute_reached_at_seconds: float
    compute_started_at_seconds: float
    compute_finished_at_seconds: float
    #: The two waits between reaching the task and starting it: inputs still
    #: being fetched, then ranges its allocations reuse still owned by a
    #: transfer. Both leave the stream idle; the boundary owns both.
    input_readiness_wait_seconds: float
    allocation_reuse_wait_seconds: float
    #: Device minus simulated, from the same invocation boundary. Entry delay
    #: and later drift both remain visible.
    start_delta_seconds: float
    end_delta_seconds: float
    #: Host clock, from invocation entry. Four instants partitioning the
    #: frontend's cycle for this task with no gap: entering the opening
    #: boundary, leaving it for the compiled call, the call returning, and
    #: leaving the closing boundary. The opening boundary, the call, and the
    #: closing boundary are the three differences between them, and the
    #: frontend's lead is `compute_reached_at_seconds` minus the second.
    before_task_entered_at_seconds: float | None
    before_task_exited_at_seconds: float | None
    after_task_entered_at_seconds: float | None
    after_task_exited_at_seconds: float | None

    def as_dict(self) -> dict[str, object]:
        """Serialize under the field names themselves, grouped by clock."""

        return {
            "execution_task_id": self.execution_task_id,
            "task_id": self.task_id,
            "execution_ordinal": self.execution_ordinal,
            "semantic_name": self.semantic_name,
            "phase": self.phase,
            "microbatch": self.microbatch,
            "simulated": {
                "simulated_ready_at_seconds": self.simulated_ready_at_seconds,
                "simulated_started_at_seconds": self.simulated_started_at_seconds,
                "simulated_finished_at_seconds": self.simulated_finished_at_seconds,
                "expected_profile_seconds": self.expected_profile_seconds,
            },
            "compute": {
                "compute_reached_at_seconds": self.compute_reached_at_seconds,
                "compute_started_at_seconds": self.compute_started_at_seconds,
                "compute_finished_at_seconds": self.compute_finished_at_seconds,
                "input_readiness_wait_seconds": self.input_readiness_wait_seconds,
                "allocation_reuse_wait_seconds": self.allocation_reuse_wait_seconds,
            },
            "delta": {
                "start_delta_seconds": self.start_delta_seconds,
                "end_delta_seconds": self.end_delta_seconds,
            },
            "host": {
                "before_task_entered_at_seconds": self.before_task_entered_at_seconds,
                "before_task_exited_at_seconds": self.before_task_exited_at_seconds,
                "after_task_entered_at_seconds": self.after_task_entered_at_seconds,
                "after_task_exited_at_seconds": self.after_task_exited_at_seconds,
            },
        }


@dataclass(frozen=True, slots=True)
class TransferRecord:
    """One scheduled transfer on its lane, simulated beside measured."""

    #: Identity: `<direction>_<sequence>`, in the lane's FIFO order.
    #: `triggered_by` is the execution task whose completion released this
    #: scheduled transfer, including control tasks; it is a key into `tasks`.
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
    #: Simulated, from invocation entry: when the transfer could
    #: start, when the lane started it, when it ended, and its priced
    #: duration at the assumed lane bandwidth.
    simulated_ready_at_seconds: float | None
    simulated_started_at_seconds: float | None
    simulated_finished_at_seconds: float | None
    #: Lane timeline, on the trace origin's axis: when the worker handed the
    #: copy to its lane, when the lane started moving bytes, and when they had
    #: landed. **Issued to started is the dependency wait**, so a copy held
    #: behind an event reads as late rather than slow. A lane whose bytes move
    #: on a stream reads the last two off timing events bracketing the copy; one
    #: whose bytes move elsewhere reads a host clock and converts through the
    #: trace's anchor. `None` where the trace could not measure this transfer.
    lane_issued_at_seconds: float | None
    lane_started_at_seconds: float | None
    lane_finished_at_seconds: float | None
    #: Device minus simulated from invocation entry; `None` without lane timing.
    start_delta_seconds: float | None
    end_delta_seconds: float | None
    #: Host clock, from invocation entry: when the action was queued,
    #: when its destination was reserved, when the worker dispatched the copy
    #: to the lane, and when the worker observed its completion. A runtime
    #: trace that ran out of room is joined to no transfer, so in that step
    #: these and the lane instants above are `None` for every transfer.
    queued_at_seconds: float | None
    reserved_at_seconds: float | None
    dispatched_at_seconds: float | None
    completion_observed_at_seconds: float | None

    def as_dict(self) -> dict[str, object]:
        """Serialize under the field names themselves, grouped by clock."""

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
                "simulated_ready_at_seconds": self.simulated_ready_at_seconds,
                "simulated_started_at_seconds": self.simulated_started_at_seconds,
                "simulated_finished_at_seconds": self.simulated_finished_at_seconds,
            },
            "lane": {
                "lane_issued_at_seconds": self.lane_issued_at_seconds,
                "lane_started_at_seconds": self.lane_started_at_seconds,
                "lane_finished_at_seconds": self.lane_finished_at_seconds,
            },
            "delta": {
                "start_delta_seconds": self.start_delta_seconds,
                "end_delta_seconds": self.end_delta_seconds,
            },
            "host": {
                "queued_at_seconds": self.queued_at_seconds,
                "reserved_at_seconds": self.reserved_at_seconds,
                "dispatched_at_seconds": self.dispatched_at_seconds,
                "completion_observed_at_seconds": self.completion_observed_at_seconds,
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
    #: intervals add up to. The effective-bandwidth field counts host
    #: copy payload bytes over complete lane time, including collective phases
    #: in sharded transfers. It is not PCIe bandwidth or full-object throughput.
    measured_transfers: int
    lane_busy_seconds: float
    effective_bandwidth_bytes_per_second: float | None
    #: The largest start delta on the lane, signed, and the transfer that
    #: reached it: where the lane had drifted furthest from the simulation.
    largest_start_delta_seconds: float | None
    largest_start_delta_transfer_id: str | None
    #: What the lane reports having moved: the counters every lane keeps in
    #: its common struct, which the runtime reads the same way for a built-in
    #: lane and for one a library registered, so nothing here knows which
    #: answered. `None` when no lane serves the direction, which is not the
    #: same as a lane reporting zero and is why this is not simply zeroed.
    #:
    #: **Cumulative, since the lane was created**, unlike every sibling field
    #: above, which describes this step. The two are different questions and
    #: neither substitutes for the other: the fields above say what this step
    #: scheduled and what the simulation expected of it, this says what the
    #: hardware has been asked for in total. Difference it across steps for a
    #: per-step figure.
    #:
    #: The duration fields are `None` unless the lane timed them. A lane that
    #: hands a transfer onward and returns never sees the moment it completed,
    #: and reporting zero there would read as "instant".
    lane_statistics: LaneStatistics | None

    def as_dict(self) -> dict[str, object]:
        return {
            "direction": self.direction,
            "transfers": self.transfers,
            "bytes": self.bytes,
            "simulated_busy_seconds": self.simulated_busy_seconds,
            "measured_transfers": self.measured_transfers,
            "lane_busy_seconds": self.lane_busy_seconds,
            "effective_bandwidth_bytes_per_second": (
                self.effective_bandwidth_bytes_per_second
            ),
            "largest_start_delta_seconds": self.largest_start_delta_seconds,
            "largest_start_delta_transfer_id": self.largest_start_delta_transfer_id,
            "lane_statistics": _lane_statistics_as_dict(self.lane_statistics),
        }


def _lane_statistics_as_dict(
    statistics: LaneStatistics | None,
) -> dict[str, object] | None:
    """Serialize what a lane reported, or `None` if it reports nothing.

    Reads the ABI structure's own field list rather than repeating it, so the
    contract is written down once: adding a counter in C adds it here.
    """

    if statistics is None:
        return None
    timed = bool(statistics.timed)
    result: dict[str, object] = {}
    for member in LaneStatistics._fields_:
        name = member[0]
        if name == "timed":
            result[name] = timed
            continue
        value = getattr(statistics, name)
        result[name] = (
            (float(value) if timed else None)
            if name.endswith("_seconds")
            else int(value)
        )
    return result


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
class TransferQueue:
    """One direction's transfers in FIFO order, with its lane's summary.

    The queue is the ordering; the lane is what moves the bytes. `order` holds
    transfer ids into this direction's group of `StepDiagnostics.transfers`,
    and an id's position is the transfer's `sequence`.
    """

    order: tuple[str, ...]
    summary: LaneSummary

    def as_dict(self) -> dict[str, object]:
        return {"summary": self.summary.as_dict(), "order": list(self.order)}


@dataclass(frozen=True, slots=True)
class Timelines:
    """The order in which one traced call ran its tasks and transfers.

    `StepDiagnostics.timelines` holds it. Its entries are ids into
    `StepDiagnostics.tasks` and `StepDiagnostics.transfers`.

    Attributes
    ----------
    host_origin_ns : int
        The host clock, ``CLOCK_MONOTONIC`` in nanoseconds, just before the
        call's origin was recorded on the compute stream. Subtracting it from
        a raw runtime event's timestamp gives the event's time from the
        origin, in nanoseconds.
    first_task_started_at_seconds : float
        When the first computational task's kernels started, in seconds on
        the device timeline.
    compute : tuple of str
        The execution task ids of the computational tasks, in compute-stream
        order. Control tasks, which only release transfers, are left out.
    fetch : TransferQueue
        The fetch lane's transfer ids in FIFO order (``order``; an id's
        position is its transfer's ``sequence``) and its ``summary``:
        transfers and bytes, simulated and measured busy time, effective
        bandwidth, largest start delta, and the lane's lifetime counters.
    evict : TransferQueue
        The evict lane, in the same form.
    """

    #: CLOCK_MONOTONIC timestamp immediately before recording the device origin.
    host_origin_ns: int
    first_task_started_at_seconds: float
    compute: tuple[str, ...]
    fetch: TransferQueue
    evict: TransferQueue

    def as_dict(self) -> dict[str, object]:
        """Write the timelines as JSON-compatible values.

        Returns
        -------
        dict
            ``clocks``, which names each clock's origin beside
            ``host_origin_ns`` and ``first_task_started_at_seconds``;
            ``compute``; and ``fetch`` and ``evict``, each with its ``order``
            and ``summary``.
        """
        return {
            "clocks": {
                "stream": "seconds from the step origin event on the device",
                "simulated": "seconds from invocation entry",
                "host": "seconds from invocation entry on the host",
                "host_origin_ns": self.host_origin_ns,
                "first_task_started_at_seconds": self.first_task_started_at_seconds,
            },
            "compute": list(self.compute),
            "fetch": self.fetch.as_dict(),
            "evict": self.evict.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class AllocatorTrace:
    """Ordered allocator lifetimes and before/after slab state."""

    events: Sequence[CapturedAllocationEvent]
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

    def as_dict(self, *, events: bool = True) -> dict[str, object]:
        """Serialize the pool's state, and the ledger unless `events` is false."""

        value: dict[str, object] = {
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
        if events:
            value["events"] = [
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
            ]
        return value


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
    began_at_ns: int
    ended_at_ns: int
    event_capacity: int
    allocation_event_capacity: int
    event_overflow: bool
    allocation_event_overflow: bool
    events: Sequence[RuntimeTraceEvent]

    def as_dict(self, *, events: bool = True) -> dict[str, object]:
        """Serialize the counters, and the raw events unless `events` is false."""

        value: dict[str, object] = {
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
            "began_at_ns": self.began_at_ns,
            "ended_at_ns": self.ended_at_ns,
            "event_capacity": self.event_capacity,
            "allocation_event_capacity": self.allocation_event_capacity,
            "event_overflow": self.event_overflow,
            "allocation_event_overflow": self.allocation_event_overflow,
        }
        if events:
            value["events"] = [item.as_dict() for item in self.events]
        return value


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
    """One traced call's measured time, reconciled with its simulation.

    `StepDiagnostics.summary` holds it. Its task sums, waits and spans cover
    the call's computational tasks, not its control tasks. The span runs from
    the first task's start to the last one's end; idle is the compute
    stream's time within it outside the tasks. Measured times are on the
    device timeline, from the call's origin event on the compute stream,
    unless named otherwise. All are in seconds.

    Attributes
    ----------
    profiled_task_seconds : float
        The tasks' profiled durations, summed.
    real_task_event_seconds : float
        The tasks' measured kernel time, from each one's start to its end,
        summed.
    task_event_delta_seconds : float
        ``real_task_event_seconds`` less ``profiled_task_seconds``.
    simulated_inter_task_idle_seconds : float
        Idle in the simulated span.
    real_inter_task_idle_seconds : float
        Idle in the measured span: ``real_inter_task_readiness_wait_seconds``
        plus ``real_inter_task_exposed_overhead_seconds``.
    inter_task_idle_delta_seconds : float
        The measured idle less the simulated idle.
    simulated_inter_task_readiness_wait_seconds : float
        The simulated stalls of the tasks after the first, summed, as
        `TaskInterval.stall_ns` counts them.
    real_inter_task_readiness_wait_seconds : float
        How long the compute stream waited at the tasks after the first, for
        their inputs and for the ranges their allocations reuse.
    real_inter_task_exposed_overhead_seconds : float
        The rest of the measured idle: time before the stream reached each
        next task, because the frontend had not yet issued it. Where the
        frontend stays ahead, this is only the timing floor between two stream
        events.
    real_initial_readiness_wait_seconds : float
        The first task's waits, which precede the span.
    real_minimum_frontend_lead_seconds : float
        The smallest lead the frontend held: for a task, the time from the
        frontend handing it to the stream until the stream reached it. Zero
        when no task has that host instant.
    simulated_selected_span_seconds : float
        The span in the simulation.
    real_selected_span_seconds : float
        The span as measured.
    selected_span_delta_seconds : float
        The measured span less the simulated span.
    simulator_makespan_seconds : float
        The whole simulated schedule, terminal transfers included.
    simulator_terminal_tail_seconds : float
        Simulated time after the last task ends.
    simulated_entry_delay_seconds : float
        Simulated time from the call's origin to the first task's start.
    real_invocation_seconds : float or None
        Measured time from the call's origin to the later of the last task's
        end and the last transfer's landing. None when a transfer has no lane
        timing.
    real_terminal_tail_seconds : float or None
        ``real_invocation_seconds`` less the last task's end; None with it.
    cycle_seconds : float or None
        Time from the call's origin to the cycle-end marker, which the compute
        stream records when the call's result is first waited for, after its
        terminal transfers have drained. None when the trace has no such
        marker.
    entry_delay_seconds : float
        Time from the call's origin to the first task's start, the first
        task's waits included.
    exposed_tail_seconds : float or None
        Time from the last task's end to the end of the cycle; None with
        ``cycle_seconds``.
    call_seconds : float
        Host-clock time `Executor.run` spent dispatching the call's tasks and
        collecting its outputs, after staging its inputs.
    optimizer_span_seconds : float
        Time from the first optimizer task's start to the last one's end; zero
        without optimizer tasks.
    phase_comparisons : tuple of PhaseTimingComparison
        One per task phase, in name order, with its ``phase``,
        ``profiled_task_seconds``, ``real_task_event_seconds`` and
        ``delta_seconds``, the measured less the profiled.
    trace_complete : bool
        False when the runtime or allocation event trace overflowed, the
        collective trace is incomplete, or a transfer has no lane timing.
    """

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
    #: here measures. Where the frontend stayed ahead it is the timing floor
    #: between two stream event records rather than a cost, so read a value
    #: near that floor as nothing and anything well above it as real.
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
    simulated_entry_delay_seconds: float
    #: Invocation entry through all computation and required terminal copies.
    #: None if a transfer lane did not provide device timestamps.
    real_invocation_seconds: float | None
    real_terminal_tail_seconds: float | None
    #: Origin to successor origin (or final marker), including caller work.
    #: None until something closes this invocation's cycle.
    cycle_seconds: float | None
    #: Origin to first computational task start, including input readiness.
    entry_delay_seconds: float
    #: Last computation to cycle end, including caller and terminal work.
    exposed_tail_seconds: float | None
    #: The whole planned call on the host clock.
    call_seconds: float
    #: First optimizer task start through last optimizer task end.
    optimizer_span_seconds: float
    phase_comparisons: tuple[PhaseTimingComparison, ...]
    trace_complete: bool

    @property
    def simulated_task_seconds(self) -> float:
        """The tasks' simulated intervals, summed, contention within them included."""
        return (
            self.simulated_selected_span_seconds
            - self.simulated_inter_task_idle_seconds
        )

    @property
    def simulated_task_contention_seconds(self) -> float:
        """``simulated_task_seconds`` less ``profiled_task_seconds``."""
        return self.simulated_task_seconds - self.profiled_task_seconds

    @property
    def task_prediction_delta_seconds(self) -> float:
        """``real_task_event_seconds`` less ``simulated_task_seconds``."""
        return self.real_task_event_seconds - self.simulated_task_seconds

    def as_dict(self) -> dict[str, object]:
        """Write the summary as JSON-compatible values.

        Returns
        -------
        dict
            Every field and property by name, with ``phase_comparisons`` as a
            mapping from each phase to its comparison's fields.
        """
        return {
            "profiled_task_seconds": self.profiled_task_seconds,
            "simulated_task_seconds": self.simulated_task_seconds,
            "simulated_task_contention_seconds": self.simulated_task_contention_seconds,
            "task_prediction_delta_seconds": self.task_prediction_delta_seconds,
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
            "simulated_entry_delay_seconds": self.simulated_entry_delay_seconds,
            "real_invocation_seconds": self.real_invocation_seconds,
            "real_terminal_tail_seconds": self.real_terminal_tail_seconds,
            "cycle_seconds": self.cycle_seconds,
            "entry_delay_seconds": self.entry_delay_seconds,
            "exposed_tail_seconds": self.exposed_tail_seconds,
            "call_seconds": self.call_seconds,
            "optimizer_span_seconds": self.optimizer_span_seconds,
            "phase_comparisons": {
                item.phase: item.as_dict() for item in self.phase_comparisons
            },
            "trace_complete": self.trace_complete,
        }


@dataclass(frozen=True, slots=True)
class StepDiagnostics:
    """What one traced call measured, beside what its plan simulated.

    `InvocationResult.diagnostics` returns it for an `Executor.run` call made
    with ``runtime_trace=True``. Times count from the call's origin, on the
    device timeline, the host clock or in the simulation, in seconds unless a
    field's name ends in ``_ns``.

    Attributes
    ----------
    summary : StepTimingSummary
        The call's measured time, reconciled with its simulation.
    tasks : mapping of str to TaskRecord
        Every task the call ran, control tasks included, by execution task id
        (``"execution_000000"`` and on, in execution order). A record holds
        the plan's task id, phase and microbatch; the task's simulated
        readiness, start and end, and its profile; when the compute stream
        reached it, how long it waited for inputs and reused ranges, and when
        its kernels started and ended; and the frontend's host-clock
        instants around it.
    transfers : TransferRecords
        Every scheduled transfer by transfer id (``"<direction>_<sequence>"``),
        in the mappings ``fetch`` and ``evict``. A record holds the task that
        released the transfer, the alias group and bytes it moved, the tasks
        that access the object before and after it and the task that last
        modified it, and its simulated, lane and host-clock instants; an
        instant the trace did not measure is None.
    timelines : Timelines
        The order of the call's tasks and of each lane's transfers.
    allocator : AllocatorTrace
        Every allocation and free the device allocator made during the call,
        in order; its pool's occupancy before and after; and its free space
        and fragmentation after.
    runtime : RuntimeTrace
        The runtime's counter changes over the call, its queues at the end,
        the trace's capacities and whether it overflowed, and its raw events.
    collectives : CollectiveTrace
        Each collective the call issued: its group, its byte counts, the task
        or transfer it belongs to, and when it was issued, started and
        finished.
    """

    summary: StepTimingSummary
    #: Every selected task by execution task id, and every scheduled transfer
    #: by transfer id grouped by direction; the timelines refer into both.
    tasks: Mapping[str, TaskRecord]
    transfers: TransferRecords
    timelines: Timelines
    allocator: AllocatorTrace
    runtime: RuntimeTrace
    collectives: CollectiveTrace = field(default_factory=CollectiveTrace)

    def as_dict(self, *, events: bool = True) -> dict[str, object]:
        """Write every view as JSON-compatible values.

        Parameters
        ----------
        events : bool, default True
            Whether to include the two event ledgers, ``allocator.events``
            and ``runtime.events``, which grow with the call.

        Returns
        -------
        dict
            Each view under its attribute's name, and ``schema``, the
            record's schema label.
        """

        return {
            "schema": artifact_schema("step_diagnostics"),
            "summary": self.summary.as_dict(),
            "tasks": {key: item.as_dict() for key, item in self.tasks.items()},
            "transfers": self.transfers.as_dict(),
            "timelines": self.timelines.as_dict(),
            "allocator": self.allocator.as_dict(events=events),
            "runtime": self.runtime.as_dict(events=events),
            "collectives": self.collectives.as_dict(),
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
    "TransferQueue",
    "TransferRecord",
    "TransferRecords",
]
