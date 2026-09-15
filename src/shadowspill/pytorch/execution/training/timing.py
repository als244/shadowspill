"""What an invocation measures about itself: the timelines every invocation
records, the armed qualification measurement, and the runtime trace behind it.
"""

from __future__ import annotations

import time
from contextlib import suppress
from typing import Any

import torch

from shadowspill.ir import MemoryAction, MemoryActionKind
from shadowspill.pytorch.diagnostics.collection import collect_step_diagnostics
from shadowspill.pytorch.diagnostics.execution import (
    StepDiagnostics,
)
from shadowspill.pytorch.diagnostics.timing import (
    ArmedExecutionTiming as _ArmedExecutionTiming,
)
from shadowspill.pytorch.diagnostics.timing import (
    ArmedTaskTiming as _ArmedTaskTiming,
)
from shadowspill.pytorch.diagnostics.timing import (
    InvocationTimelines,
    InvocationTiming,
)
from shadowspill.pytorch.lowering.training import (
    TrainingTaskEntrypoint,
)
from shadowspill.pytorch.runtime_adapter.bridge import (
    RuntimeBridge,
    begin_runtime_trace,
    end_and_read_runtime_trace,
    prepare_runtime_trace,
    statistics,
)

from ..records import (
    PlanRun as _PlanRun,
)
from .values import alias_accesses


class ExecutionTiming:
    """The timing and tracing state of one training executor.

    Every invocation records where it begins, where its first task starts and
    where its last task ends on the compute stream; the events are created here
    so no invocation creates one. The armed measurement is qualification-only
    and default-off; the runtime trace behind it is allocated lazily.
    """

    def __init__(self, bridge: RuntimeBridge, task_ids: tuple[str, ...]) -> None:
        self._bridge = bridge
        self._task_ids = task_ids
        self.armed: _ArmedExecutionTiming | None = None
        self.prior_invocation_drain_ns = 0
        timing_event_factory: Any = torch.cuda.Event
        self._timelines = InvocationTimelines(
            lambda: timing_event_factory(enable_timing=True)
        )
        # Detailed tracing is default-off and allocated lazily. Full-model
        # schedules emit several records per action plus readiness and
        # retirement records, so task count alone is not a safe bound. Keep a
        # deliberately generous fixed capacity and report any overflow in the
        # public reconciliation summary rather than truncating silently.
        self._trace_allocation_capacity = 1_000_000
        self._trace_event_capacity = 1_000_000
        self._trace_start_event: torch.cuda.Event | None = None
        self._trace_end_event: torch.cuda.Event | None = None
        self._trace_origin_event: torch.cuda.Event | None = None
        self._trace_task_events: dict[
            str,
            tuple[
                torch.cuda.Event,
                torch.cuda.Event,
                torch.cuda.Event,
                torch.cuda.Event,
            ],
        ] = {}

    @property
    def span_pending(self) -> bool:
        return self._timelines.span_pending

    def begin_invocation(self, step_number: int, stream: torch.cuda.Stream) -> Any:
        """Open the invocation's timeline; the caller numbers the step."""

        return self._timelines.begin(step_number, stream)

    def prepare(self) -> None:
        """Lazily allocate reusable trace buffers and timing events."""

        prepare_runtime_trace(
            self._bridge,
            event_capacity=self._trace_event_capacity,
            allocation_event_capacity=self._trace_allocation_capacity,
        )
        event_factory: Any = torch.cuda.Event
        task_ids = self._task_ids
        self._trace_origin_event = event_factory(enable_timing=True)
        self._trace_start_event = event_factory(enable_timing=True)
        self._trace_end_event = event_factory(enable_timing=True)
        self._trace_task_events = {
            task_id: (
                event_factory(enable_timing=True),
                event_factory(enable_timing=True),
                event_factory(enable_timing=True),
                event_factory(enable_timing=True),
            )
            for task_id in task_ids
        }
        # PyTorch creates CUDA event handles lazily on first record. Force that
        # one-time setup before the real trace begins, then reuse every event.
        stream = torch.cuda.current_stream()
        self._trace_origin_event.record(stream)
        self._trace_start_event.record(stream)
        self._trace_end_event.record(stream)
        for events in self._trace_task_events.values():
            for event in events:
                event.record(stream)
        stream.synchronize()

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
            self._bridge,
            step_id=step_number,
            origin_event_handle=int(timing.origin_event.cuda_event),
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

        self._timelines.mark_end(torch.cuda.current_stream())

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
        stream = timing.stream or torch.cuda.current_stream()
        stream.synchronize()
        with suppress(BaseException):
            end_and_read_runtime_trace(self._bridge)
        self.armed = None

    def record_compute_start(self, stream: torch.cuda.Stream | None) -> None:
        if stream is None:
            return
        self._timelines.start_span(stream)
        timing = self.armed
        if timing is None or timing.started:
            return
        timing.start_event.record(stream)
        timing.started = True
        timing.stream = stream

    def record_compute_end(self, stream: torch.cuda.Stream | None) -> None:
        if stream is None:
            return
        self._timelines.end_span(stream)
        timing = self.armed
        if timing is None or timing.finished:
            return
        timing.end_event.record(stream)
        timing.finished = True

    def begin_task(self, entrypoint: TrainingTaskEntrypoint) -> _ArmedTaskTiming | None:
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
                raise AssertionError("task timing omitted its CUDA stream")
            task.readiness_event.record(stream)

    @staticmethod
    def record_task_inputs_ready(
        task: _ArmedTaskTiming | None, stream: torch.cuda.Stream | None
    ) -> None:
        """Mark where waiting for inputs ends and waiting for ranges begins."""

        if task is not None:
            if stream is None:
                raise AssertionError("task timing omitted its CUDA stream")
            task.inputs_ready_event.record(stream)

    @staticmethod
    def record_task_start(
        task: _ArmedTaskTiming | None, stream: torch.cuda.Stream | None
    ) -> None:
        if task is not None:
            if stream is None:
                raise AssertionError("task timing omitted its CUDA stream")
            task.start_event.record(stream)

    @staticmethod
    def record_task_end(
        task: _ArmedTaskTiming | None, stream: torch.cuda.Stream | None
    ) -> None:
        if task is not None:
            if stream is None:
                raise AssertionError("task timing omitted its CUDA stream")
            task.end_event.record(stream)


__all__ = ["ExecutionTiming"]
