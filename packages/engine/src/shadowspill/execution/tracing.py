"""Opt-in task and transfer recording for one saved-Plan invocation."""

import time

import torch

from shadowspill.diagnostics.step import StepDiagnostics
from shadowspill.execution.recording.spec import (
    TracedInvocation,
    TracedTask,
    alias_accesses,
)
from shadowspill.execution.recording.timing import ExecutionTiming
from shadowspill.runtime.plan.tracing import trace_capacities

from .paths import ExecutionPath


class InvocationTrace:
    def __init__(self, path: ExecutionPath):
        plan = path.plan
        self.result: StepDiagnostics | None = None
        self.timing = ExecutionTiming(
            path.bridge,
            tuple(task.entry.task_id for task in path.tasks),
            trace_capacities(
                ((task.task, task.memory_envelope) for task in path.tasks),
                plan.schedule.actions,
            ),
        )
        profiles = {
            profile.profile_id: profile for profile in plan._costed_program.profiles
        }
        names = plan.program.bindings.get("task_names", {})
        self.description = TracedInvocation(
            tasks=tuple(
                TracedTask(
                    bound.entry,
                    profiles[bound.task.profile_id].runtime_ns / 1e9,
                    ordinal,
                    names.get(bound.entry.task_id, bound.entry.task_id),
                )
                for ordinal, bound in enumerate(path.tasks)
            ),
            actions=plan.schedule.actions,
            simulation=plan.timeline,
            alias_accesses=alias_accesses(
                plan._costed_program,
                ((ordinal, bound.task) for ordinal, bound in enumerate(path.tasks)),
            ),
        )
        numerical = [b.entry.task_id for b in path.tasks if b.task.phase != "control"]
        self.first, self.last = numerical[0], numerical[-1]

    def start(self) -> None:
        self.timing.arm(self.description)
        armed = self.timing.armed
        assert armed is not None
        stream = torch.cuda.current_stream()
        armed.timeline = self.timing.begin_invocation(0, stream)
        self.timing.record_origin(stream)
        self.timing.begin_armed_runtime_trace(armed, 0)
        armed.dispatch_call_started_ns = time.perf_counter_ns()

    def dispatched(self) -> None:
        assert self.timing.armed is not None
        self.timing.armed.dispatch_call_finished_ns = time.perf_counter_ns()

    def finish(self) -> None:
        try:
            self.timing.mark_cycle_end()
            self.result = self.timing.collect_step_diagnostics()
        finally:
            self.timing.release()

    def cancel(self) -> None:
        try:
            self.timing.cancel()
        finally:
            self.timing.release()
