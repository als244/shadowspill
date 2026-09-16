"""What an invocation measures about itself: the timelines every invocation
records, the armed qualification measurement, and the runtime trace behind it.
"""

from __future__ import annotations

import time
from contextlib import suppress
from typing import Any

import torch

from shadowspill.diagnostics.collection import collect_step_diagnostics
from shadowspill.diagnostics.step import (
    StepDiagnostics,
)
from shadowspill.diagnostics.timing import (
    ArmedExecutionTiming as _ArmedExecutionTiming,
)
from shadowspill.diagnostics.timing import (
    ArmedTaskTiming as _ArmedTaskTiming,
)
from shadowspill.diagnostics.timing import (
    InvocationTimelines,
    InvocationTiming,
)
from shadowspill.ir import MemoryAction, MemoryActionKind
from shadowspill.runtime.plan import (
    RuntimeBridge,
    begin_runtime_trace,
    end_and_read_runtime_trace,
    prepare_runtime_trace,
    statistics,
)
from shadowspill.runtime.timing import Marker, wait_for_stream
from shadowspill.task.entrypoints import TaskEntrypoint

from ..records import (
    PlanRun as _PlanRun,
)
from .values import alias_accesses


class ExecutionTiming:
    """The timing and tracing state of one training executor.

    Every invocation records where it begins, where its first task starts and
    where its last task ends on the compute stream. Every instant is a marker
    the runtime holds, taken once here and recorded again each invocation, so
    the executor and the runtime's own transfers share one clock. The armed
    measurement is qualification-only and default-off; the runtime trace behind
    it is allocated lazily.
    """

    def __init__(self, bridge: RuntimeBridge, task_ids: tuple[str, ...]) -> None:
        self._bridge = bridge
        self._task_ids = task_ids
        self.armed: _ArmedExecutionTiming | None = None
        self.prior_invocation_drain_ns = 0
        self._runtime_handle = bridge.runtime._runtime_handle
        self._timelines = InvocationTimelines(self._runtime_handle)
        # Detailed tracing is default-off and allocated lazily. Full-model
        # schedules emit several records per action plus readiness and
        # retirement records, so task count alone is not a safe bound. Keep a
        # deliberately generous fixed capacity and report any overflow in the
        # public reconciliation summary rather than truncating silently.
        self._trace_allocation_capacity = 1_000_000
        self._trace_event_capacity = 1_000_000
        self._trace_start_event: Marker | None = None
        self._trace_end_event: Marker | None = None
        self._trace_origin_event: Marker | None = None
        self._trace_task_events: dict[str, tuple[Marker, Marker, Marker, Marker]] = {}

    @property
    def span_pending(self) -> bool:
        return self._timelines.span_pending

    def begin_invocation(self, step_number: int, stream: torch.cuda.Stream) -> Any:
        """Open the invocation's timeline; the caller numbers the step."""

        return self._timelines.begin(step_number, _handle(stream))

    def prepare(self) -> None:
        """Lazily allocate reusable trace buffers and timing markers.

        The runtime's markers hold real backend events from the moment they are
        taken, so nothing is warmed up here: the first instant each one records
        is a measurement.
        """

        # The markers come first. Preparing a trace reserves the timing pool a
        # floor of free leases for the lanes it measures, and leases already
        # held do not count against it -- so taking the markers first leaves
        # that floor intact, where taking them afterwards would spend it.
        marker = self._marker
        self._trace_origin_event = marker()
        self._trace_start_event = marker()
        self._trace_end_event = marker()
        self._trace_task_events = {
            task_id: (marker(), marker(), marker(), marker())
            for task_id in self._task_ids
        }
        prepare_runtime_trace(
            self._bridge,
            event_capacity=self._trace_event_capacity,
            allocation_event_capacity=self._trace_allocation_capacity,
        )

    def _marker(self) -> Marker:
        return Marker(self._runtime_handle)

    def release(self) -> None:
        """Give every marker back to the runtime; this executor is done.

        Markers grow the timing pool when its reserve is spent, so an executor
        that is replaced rather than closed would leave the pool larger every
        time. Released after the plan is idle, so nothing is still recording.
        """

        self._timelines.release()
        for marker in (
            self._trace_origin_event,
            self._trace_start_event,
            self._trace_end_event,
        ):
            if marker is not None:
                marker.release()
        self._trace_origin_event = None
        self._trace_start_event = None
        self._trace_end_event = None
        for markers in self._trace_task_events.values():
            for marker in markers:
                marker.release()
        self._trace_task_events = {}

    def arm(self, run: _PlanRun, *, trace_setup_ns: int = 0) -> None:
        """Bracket the next invocation's numerical compute stream only.

        This qualification-only measurement begins after the first task's
        readiness waits and ends after the final optimizer launch. It excludes
        invocation-start staging and terminal writeback without synchronizing
        ordinary execution.
        """

        if self.armed is not None:
            raise RuntimeError("a compute timing measurement is already armed")
        if (
            self._trace_origin_event is None
            or self._trace_start_event is None
            or self._trace_end_event is None
        ):
            self.prepare()
        origin_event = self._trace_origin_event
        start_event = self._trace_start_event
        end_event = self._trace_end_event
        if origin_event is None or start_event is None or end_event is None:
            raise AssertionError("trace event preparation did not complete")
        tasks = {
            record.task.task_id: _ArmedTaskTiming(
                record.entrypoint,
                run.expected_task_seconds[record.task.task_id],
                record.execution_ordinal,
                record.semantic_name,
                *self._trace_task_events[record.task.task_id],
            )
            for record in run.execution
        }
        armed = _ArmedExecutionTiming(
            origin_event,
            start_event,
            end_event,
            tasks,
            tuple(record.task.task_id for record in run.execution),
            actions=(
                tuple(
                    MemoryAction("task_000000", alias_id, MemoryActionKind.FETCH)
                    for alias_id in run.initial_fetches
                )
                + run.plan.schedule.actions
            ),
            simulation=run.simulation,
            trace_setup_ns=trace_setup_ns,
            alias_accesses=alias_accesses(run),
        )
        self.armed = armed

    def begin_armed_runtime_trace(
        self, timing: _ArmedExecutionTiming, step_number: int
    ) -> None:
        """Open the current invocation's runtime trace after prior work is idle."""

        timing.statistics_before = statistics(self._bridge)
        # Transfers are measured on their lanes from the same origin event
        # the compute-stream markers use, so every lane shares one timeline.
        begin_runtime_trace(
            self._bridge, step_id=step_number, origin=timing.origin_event
        )

    @property
    def prior_invocation_drain_seconds(self) -> float:
        """How long the last call waited for the previous invocation to drain."""

        return self.prior_invocation_drain_ns / 1e9

    def mark_cycle_end(self) -> None:
        """Close the current invocation's cycle where the next one would begin.

        Recorded on the compute stream behind everything the invocation
        enqueued, so its cycle reads the same as if another invocation had
        followed it at once. A loop that measures its last step calls this
        after that step's call returns.
        """

        self._timelines.mark_end(_handle(torch.cuda.current_stream()))

    def invocation_timings(self) -> tuple[InvocationTiming, ...]:
        """Every invocation whose cycle is complete, once each, oldest first.

        An invocation's cycle is complete once a later invocation began or
        `mark_cycle_end()` closed it. Reading waits for the device to reach
        the closing event, and for nothing else.
        """

        return self._timelines.drain()

    def collect_step_diagnostics(self) -> StepDiagnostics:
        """Synchronize and resolve the structured trace for one real call."""

        timing = self.armed
        if timing is None:
            raise RuntimeError("no execution timing measurement is armed")
        try:
            return collect_step_diagnostics(timing, self._bridge)
        finally:
            self.armed = None

    def cancel(self) -> None:
        """Synchronously tear down an armed debug trace after execution failure."""

        timing = self.armed
        if timing is None:
            return
        stream = timing.stream or _handle(torch.cuda.current_stream())
        wait_for_stream(self._runtime_handle, stream)
        with suppress(BaseException):
            end_and_read_runtime_trace(self._bridge)
        self.armed = None

    def record_origin(self, stream: torch.cuda.Stream) -> None:
        """Record where the armed invocation begins, if one is armed."""

        timing = self.armed
        if timing is not None:
            timing.origin_event.record(_handle(stream))

    def record_compute_start(self, stream: torch.cuda.Stream | None) -> None:
        if stream is None:
            return
        handle = _handle(stream)
        self._timelines.start_span(handle)
        timing = self.armed
        if timing is None or timing.started:
            return
        timing.start_event.record(handle)
        timing.started = True
        timing.stream = handle

    def record_compute_end(self, stream: torch.cuda.Stream | None) -> None:
        if stream is None:
            return
        self._timelines.end_span(_handle(stream))
        timing = self.armed
        if timing is None or timing.finished:
            return
        timing.end_event.record(_handle(stream))
        timing.finished = True

    def begin_task(self, entrypoint: TaskEntrypoint) -> _ArmedTaskTiming | None:
        timing = self.armed
        if timing is None:
            return None
        task = timing.tasks[entrypoint.task_id]
        task.before_task_enter_ns = time.perf_counter_ns()
        return task

    @staticmethod
    def finish_task(task: _ArmedTaskTiming | None) -> None:
        if task is None:
            return
        task.after_task_exit_ns = time.perf_counter_ns()

    @staticmethod
    def record_task_readiness(
        task: _ArmedTaskTiming | None, stream: torch.cuda.Stream | None
    ) -> None:
        if task is not None:
            if stream is None:
                raise AssertionError("task timing omitted its compute stream")
            task.readiness_event.record(_handle(stream))

    @staticmethod
    def record_task_inputs_ready(
        task: _ArmedTaskTiming | None, stream: torch.cuda.Stream | None
    ) -> None:
        """Mark where waiting for inputs ends and waiting for ranges begins."""

        if task is not None:
            if stream is None:
                raise AssertionError("task timing omitted its compute stream")
            task.inputs_ready_event.record(_handle(stream))

    @staticmethod
    def record_task_start(
        task: _ArmedTaskTiming | None, stream: torch.cuda.Stream | None
    ) -> None:
        if task is not None:
            if stream is None:
                raise AssertionError("task timing omitted its compute stream")
            task.start_event.record(_handle(stream))

    @staticmethod
    def record_task_end(
        task: _ArmedTaskTiming | None, stream: torch.cuda.Stream | None
    ) -> None:
        if task is not None:
            if stream is None:
                raise AssertionError("task timing omitted its compute stream")
            task.end_event.record(_handle(stream))


def _handle(stream: torch.cuda.Stream) -> int:
    """The integer the runtime names this stream by."""

    return int(stream.cuda_stream)


__all__ = ["ExecutionTiming"]
