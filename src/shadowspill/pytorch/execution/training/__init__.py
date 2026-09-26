"""Exact accumulated-training dispatch through selected AOT graph pairs.

`TrainingExecutor` runs the step's plan on every invocation. Its parts are the
modules beside it: `admission` admits a run to the runtime; `boundary` and
`publication` are the two halves of every task boundary, as functions over the
executor; `timing` is what an invocation measures about itself; and
`optimizer_state` is the optimizer's state as the plan holds it, which a
checkpoint reads and writes.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from shadowspill.diagnostics.timing import (
    ArmedExecutionTiming as _ArmedExecutionTiming,
)
from shadowspill.ir import ExecutionPlan, MemoryAction, MemoryActionKind
from shadowspill.pytorch.invocation import ReusableCompletionEvent
from shadowspill.pytorch.lowering.training import LoweredTrainingProgram
from shadowspill.pytorch.materialization.training import TrainingMaterializedState
from shadowspill.pytorch.runtime_adapter.boundaries import (
    acquire_for_caller,
    submit_initial_actions,
    transfer_outputs_to_caller,
)
from shadowspill.runtime.fixed_layout import RuntimeFixedLayout
from shadowspill.runtime.plan import (
    RuntimeBridge,
    TaskMemoryEnvelope,
    clear_tasks,
)
from shadowspill.simulator import SimulationResult

from ..annotations import AnnotatedExecutor, TaskBoundaryAnnotations
from ..records import (
    PlanRun as _PlanRun,
)
from ..records import (
    build_plan_run,
)
from ..timing import ExecutionTiming
from .admission import admit_run
from .boundary import execute_task
from .optimizer_state import OptimizerState


class TrainingExecutor(AnnotatedExecutor):
    """Execute selected forward/backward variants and one optimizer update."""

    def __init__(
        self,
        lowered: LoweredTrainingProgram,
        plan: ExecutionPlan,
        bridge: RuntimeBridge,
        state: TrainingMaterializedState,
        functions: dict[str, Callable[..., object]],
        optimizer: torch.optim.Optimizer,
        *,
        optimizer_parameters: Mapping[str, torch.nn.Parameter],
        simulation: SimulationResult,
        fixed_layout: RuntimeFixedLayout,
        memory_envelopes: Mapping[str, TaskMemoryEnvelope],
    ) -> None:
        self._bridge = bridge
        self._state = state
        self._functions = functions
        run = build_plan_run(
            lowered,
            plan,
            simulation=simulation,
            bridge=bridge,
            functions=functions,
            memory_envelopes=memory_envelopes,
        )
        self.optimizer_state = OptimizerState(
            optimizer, state, bridge, lowered, optimizer_parameters
        )
        # Materialization uses a short-lived action batch. Replace it with the
        # step's plan, admitted once for every invocation.
        clear_tasks(self._bridge)
        self._run = admit_run(self._bridge, run, fixed_layout)
        self._gradients = {
            state.bridge.objects.alias_for_object(
                item.gradient_object_id
            ): model_parameter
            for item in lowered.gradients
            for model_parameter in (state.model.get_parameter(item.parameter_name),)
        }
        self._invocations = 0
        self.timing = ExecutionTiming(
            bridge,
            tuple(dict.fromkeys(record.task.task_id for record in self._run.execution)),
        )
        self._task_annotations = TaskBoundaryAnnotations(self._bridge)
        self._completion = ReusableCompletionEvent(
            bridge.runtime._runtime_handle, state.device
        )

    @property
    def run(self) -> _PlanRun:
        """The plan every invocation runs."""

        return self._run

    def release_timing(self) -> None:
        """Give every marker this executor holds back to the runtime."""

        self._completion.release()
        self.timing.release()

    def __call__(
        self, inputs: Sequence[Sequence[Any]], step_number: int
    ) -> tuple[tuple[torch.Tensor, ...], tuple[Any, ...]]:
        timing = self.timing.armed
        run = self._begin_invocation(inputs, timing, step_number)
        self._submit_initial_placement(run, timing)
        ordered = self._execute_program(run)
        self._handoff_public_outputs(run, ordered)
        losses, metrics = self._rebuild_objective_results(ordered)
        self._invocations += 1
        if timing is not None:
            timing.dispatch_call_finished_ns = time.perf_counter_ns()
        return losses, metrics

    def _begin_invocation(
        self,
        inputs: Sequence[Sequence[Any]],
        timing: _ArmedExecutionTiming | None,
        step_number: int,
    ) -> _PlanRun:
        stream = torch.cuda.current_stream()
        if timing is not None:
            timing.dispatch_call_started_ns = time.perf_counter_ns()
        self.timing.record_origin(stream)
        # The caller numbers the step, so the timings carry the count a
        # restored checkpoint resumed from. `_invocations` counts this
        # process's calls and only decides whether there is a prior plan to
        # drain.
        timeline = self.timing.begin_invocation(step_number, stream)
        if timing is not None:
            timing.timeline = timeline
        self.timing.prior_invocation_drain_ns = 0
        if self._invocations:
            # V1 plans have a fresh terminal state. Preserve asynchronous
            # StepResult construction, but do not accidentally overlap the
            # next invocation with terminal transfers from the prior plan.
            #
            # Timed on every invocation rather than only a traced one: the
            # first invocation has nothing to wait for, so a trace taken on a
            # warm first step is exactly the step that never pays this.
            started_ns = time.perf_counter_ns()
            self._bridge.wait_until_idle()
            self.timing.prior_invocation_drain_ns = time.perf_counter_ns() - started_ns
            if timing is not None:
                timing.prior_invocation_drain_ns = self.timing.prior_invocation_drain_ns
        if timing is not None:
            self.timing.begin_armed_runtime_trace(timing, step_number)
        # Staging the inputs waits for the whole runtime, so every earlier
        # call has drained here and must have left the layout empty.
        self._state.refresh_inputs(inputs)
        self._bridge.require_empty_layout()
        return self._run

    def _submit_initial_placement(
        self,
        run: _PlanRun,
        timing: _ArmedExecutionTiming | None,
    ) -> None:
        started_ns = time.perf_counter_ns() if timing is not None else 0
        with self._task_annotations.range("shadowspill.training.initial_actions"):
            if run.initial_task_id is None:
                raise AssertionError("run has no admitted initial-placement task")
            submit_initial_actions(
                self._bridge,
                tuple(
                    MemoryAction("task_000000", alias_id, MemoryActionKind.FETCH)
                    for alias_id in run.initial_fetches
                ),
                task_number=run.initial_task_id,
            )
        if timing is not None:
            timing.dispatch_initial_actions_ns = time.perf_counter_ns() - started_ns

    def _execute_program(
        self,
        run: _PlanRun,
    ) -> tuple[tuple[torch.Tensor, ...], ...]:
        public_tensors: dict[int, tuple[torch.Tensor, ...]] = {}
        for record in run.execution:
            entrypoint = record.entrypoint
            outputs = execute_task(self, run, record)
            if (
                entrypoint.options.phase == "forward"
                and entrypoint.options.repetition is not None
            ):
                public_tensors[entrypoint.options.repetition] = outputs[
                    : entrypoint.options.public_output_count
                ]
        return tuple(public_tensors[index] for index in range(len(public_tensors)))

    def _handoff_public_outputs(
        self,
        run: _PlanRun,
        ordered: tuple[tuple[torch.Tensor, ...], ...],
    ) -> None:
        aliases = tuple(
            alias_id for values in run.public_by_microbatch for alias_id in values
        )
        tensors = tuple(tensor for values in ordered for tensor in values)
        bindings = acquire_for_caller(
            self._bridge,
            aliases,
            tensors,
            acquisition_handle=run.caller_acquisition_handle,
        )
        transfer_outputs_to_caller(
            self._bridge,
            aliases,
            tensors,
            bindings,
            acquisition_handle=run.caller_acquisition_handle,
        )
        for alias_id in aliases:
            self._state.forget(alias_id, run.object_ids_by_alias.get(alias_id, ()))

    def _rebuild_objective_results(
        self,
        ordered: tuple[tuple[torch.Tensor, ...], ...],
    ) -> tuple[tuple[torch.Tensor, ...], tuple[Any, ...]]:
        losses: list[torch.Tensor] = []
        metrics: list[Any] = []
        for capture, values in zip(self._state.captures, ordered, strict=True):
            losses.append(values[0].detach())
            metrics.append(
                capture.objective_schema.rebuild_metrics(
                    tuple(value.detach() for value in values[1:])
                )
            )
        return tuple(losses), tuple(metrics)


__all__ = ["TrainingExecutor"]
