"""Mutable timing state used only while one execution trace is armed."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from shadowspill.ir import MemoryAction
from shadowspill.planner.diagnostics.mapping import FrozenMapping
from shadowspill.pytorch.lowering.training import TrainingTaskEntrypoint
from shadowspill.pytorch.runtime_adapter.abi import AdapterStatistics
from shadowspill.simulator import SimulationResult


@dataclass(slots=True)
class ArmedTaskTiming:
    """Reusable event and host-clock state for one execution task."""

    entrypoint: TrainingTaskEntrypoint
    expected_profile_seconds: float
    execution_ordinal: int
    semantic_name: str
    readiness_event: torch.cuda.Event
    inputs_ready_event: torch.cuda.Event
    start_event: torch.cuda.Event
    end_event: torch.cuda.Event
    #: Four instants that partition one task's frontend cycle with no gap:
    #: entering the opening boundary, leaving it for the compiled call, the
    #: call returning, and leaving the closing boundary. Every duration
    #: between boundaries is a difference of two of these, so none is stored.
    before_task_enter_ns: int = 0
    before_task_exit_ns: int = 0
    after_task_enter_ns: int = 0
    after_task_exit_ns: int = 0
    dispatch_input_lookup_ns: int = 0
    dispatch_storage_rebind_ns: int = 0
    dispatch_input_acquire_ns: int = 0
    dispatch_allocation_reuse_ns: int = 0
    dispatch_argument_assembly_ns: int = 0
    dispatch_output_flatten_ns: int = 0
    dispatch_output_classification_ns: int = 0
    dispatch_output_adoption_ns: int = 0
    dispatch_output_state_publish_ns: int = 0
    dispatch_output_publish_ns: int = 0
    dispatch_dematerialize_ns: int = 0
    dispatch_cleanup_ns: int = 0


@dataclass(slots=True)
class ArmedExecutionTiming:
    """Mutable trace state spanning one complete planned invocation."""

    origin_event: torch.cuda.Event
    start_event: torch.cuda.Event
    end_event: torch.cuda.Event
    tasks: dict[str, ArmedTaskTiming]
    task_order: tuple[str, ...]
    started: bool = False
    finished: bool = False
    dispatch_call_started_ns: int = 0
    dispatch_call_finished_ns: int = 0
    prior_invocation_drain_ns: int = 0
    dispatch_initial_actions_ns: int = 0
    stream: torch.cuda.Stream | None = None
    statistics_before: AdapterStatistics | None = None
    actions: tuple[MemoryAction, ...] = ()
    #: The invocation's always-on timeline, whose cycle the summary reports
    #: when a successor exists by the time the trace is resolved.
    timeline: InvocationTimeline | None = None
    simulation: SimulationResult | None = None
    trace_setup_ns: int = 0
    #: Every read and write of each alias group by the selected tasks, as
    #: (execution ordinal, is_write) in execution order: what a transfer's
    #: record uses to name the task whose result it carries and the task
    #: that will read it.
    alias_accesses: Mapping[str, tuple[tuple[int, bool], ...]] = field(
        default_factory=lambda: FrozenMapping({})
    )


@dataclass(slots=True)
class InvocationTimeline:
    """Three timing events one invocation records on the compute stream.

    `origin` is recorded where the invocation begins on the stream, before
    its first task; `span_start` where its first task's compute starts and
    `span_end` where its last task's compute ends; `successor` is the next
    invocation's origin, or the marker `mark_cycle_end()` records, whichever
    the stream reaches first after this invocation. The invocation's cycle is
    origin to successor, so it is known only once a successor exists.
    """

    origin: torch.cuda.Event
    span_start: torch.cuda.Event
    span_end: torch.cuda.Event
    step_number: int = 0
    successor: torch.cuda.Event | None = None
    started: bool = False
    finished: bool = False


@dataclass(frozen=True, slots=True)
class InvocationTiming:
    """One completed invocation on the device clock: what its step cost.

    The four parts partition the cycle exactly:
    ``cycle_seconds == opening_delay_seconds + selected_span_seconds +
    exposed_tail_seconds``. The opening delay is the stream's wait from the
    origin to the first task's compute, which is the first task's readiness waits and
    whatever the opening restore and staging still held it for. The span is
    first task start to last task end. The exposed tail is the stream time
    after the last task before the next invocation's origin (or the end
    marker): terminal work the stream still had to do, not the transfers that
    drained on the lanes meanwhile, which cost the step nothing.
    """

    step_number: int
    cycle_seconds: float
    opening_delay_seconds: float
    selected_span_seconds: float
    exposed_tail_seconds: float

    def as_dict(self) -> dict[str, object]:
        return {
            "step_number": self.step_number,
            "cycle_seconds": self.cycle_seconds,
            "opening_delay_seconds": self.opening_delay_seconds,
            "selected_span_seconds": self.selected_span_seconds,
            "exposed_tail_seconds": self.exposed_tail_seconds,
        }


class InvocationTimelines:
    """The timelines of recent invocations, reused round-robin.

    `begin()` gives an invocation its timeline and makes its origin the
    successor of the previous one; `mark_end()` records a marker that
    completes the current invocation's cycle where the next invocation's
    origin would; `drain()` returns every invocation whose cycle is complete,
    once each. Timelines are reused only after they were drained, so an event
    is never re-recorded while a reading of it is pending; a caller that never
    drains loses the oldest timelines past `capacity`, never a running one.
    """

    def __init__(
        self,
        event_factory: Callable[[], Any],
        *,
        capacity: int = 16,
    ) -> None:
        if capacity < 2:
            raise ValueError("invocation timeline capacity must be at least 2")
        self._event_factory = event_factory
        self._capacity = capacity
        self._timelines: list[InvocationTimeline] = []
        self._pending: list[InvocationTimeline] = []
        self._free: list[InvocationTimeline] = []
        self._current: InvocationTimeline | None = None
        self._end_marker: Any = None

    @property
    def current(self) -> InvocationTimeline | None:
        return self._current

    @property
    def span_pending(self) -> bool:
        """Whether the running invocation has yet to record its first task's start."""

        return self._current is not None and not self._current.started

    def _acquire(self) -> InvocationTimeline:
        if self._free:
            return self._free.pop()
        if len(self._timelines) < self._capacity:
            timeline = InvocationTimeline(
                self._event_factory(), self._event_factory(), self._event_factory()
            )
            self._timelines.append(timeline)
            return timeline
        # Every timeline is pending: the caller has not drained. Drop the
        # oldest completed one rather than touching the running invocation.
        dropped = self._pending.pop(0)
        return dropped

    def begin(self, step_number: int, stream: Any) -> InvocationTimeline:
        """Record where an invocation begins on the stream; return its timeline."""

        timeline = self._acquire()
        timeline.step_number = step_number
        timeline.successor = None
        timeline.started = False
        timeline.finished = False
        timeline.origin.record(stream)
        previous = self._current
        if previous is not None and previous.successor is None:
            previous.successor = timeline.origin
        self._current = timeline
        self._pending.append(timeline)
        return timeline

    def start_span(self, stream: Any) -> None:
        timeline = self._current
        if timeline is not None and not timeline.started:
            timeline.span_start.record(stream)
            timeline.started = True

    def end_span(self, stream: Any) -> None:
        timeline = self._current
        if timeline is not None and not timeline.finished:
            timeline.span_end.record(stream)
            timeline.finished = True

    def mark_end(self, stream: Any) -> None:
        """Complete the current invocation's cycle where the next one would begin."""

        timeline = self._current
        if timeline is None or timeline.successor is not None:
            return
        if self._end_marker is None:
            self._end_marker = self._event_factory()
        self._end_marker.record(stream)
        timeline.successor = self._end_marker

    def drain(self) -> tuple[InvocationTiming, ...]:
        """Every invocation whose cycle is complete, once each, oldest first."""

        done: list[InvocationTiming] = []
        keep: list[InvocationTimeline] = []
        for timeline in self._pending:
            successor = timeline.successor
            if successor is None or not (timeline.started and timeline.finished):
                keep.append(timeline)
                continue
            successor.synchronize()
            done.append(
                InvocationTiming(
                    step_number=timeline.step_number,
                    cycle_seconds=float(timeline.origin.elapsed_time(successor)) / 1e3,
                    opening_delay_seconds=(
                        float(timeline.origin.elapsed_time(timeline.span_start)) / 1e3
                    ),
                    selected_span_seconds=(
                        float(timeline.span_start.elapsed_time(timeline.span_end))
                        / 1e3
                    ),
                    exposed_tail_seconds=(
                        float(timeline.span_end.elapsed_time(successor)) / 1e3
                    ),
                )
            )
            if timeline is self._current:
                # Its cycle is closed by the end marker; a later invocation
                # starts a new timeline rather than reusing this one.
                self._current = None
            self._free.append(timeline)
        self._pending = keep
        return tuple(done)


__all__ = [
    "ArmedExecutionTiming",
    "ArmedTaskTiming",
    "InvocationTimeline",
    "InvocationTimelines",
    "InvocationTiming",
]
