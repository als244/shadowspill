"""Exact accumulated-training dispatch through selected AOT graph pairs.

`TrainingExecutor` runs one plan per invocation: the initial plan while a lazy
optimizer's state does not exist yet, the recurrent plan after. Its parts are
the modules beside it: `admission` admits a run to the runtime; `boundary` and
`publication` are the two halves of every task boundary, as functions over the
executor; `timing` is what an invocation measures about itself; and
`optimizer_state` is the optimizer's state as the plan holds it, which a
checkpoint reads and writes.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

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
        initial: tuple[LoweredTrainingProgram, ExecutionPlan] | None,
        recurrent: tuple[LoweredTrainingProgram, ExecutionPlan],
        bridge: RuntimeBridge,
        state: TrainingMaterializedState,
        functions: dict[str, Callable[..., object]],
        optimizer: torch.optim.Optimizer,
        *,
        recurrent_simulation: SimulationResult,
        initial_simulation: SimulationResult | None = None,
        initial_fixed_layout: RuntimeFixedLayout | None = None,
        recurrent_fixed_layout: RuntimeFixedLayout,
        initial_memory_envelopes: Mapping[str, TaskMemoryEnvelope] | None = None,
        recurrent_memory_envelopes: Mapping[str, TaskMemoryEnvelope],
        optimizer_state_preinitialized: bool = False,
        optimizer_state_was_lazy: bool = False,
    ) -> None:
        self._bridge = bridge
        self._state = state
        self._functions = functions
        if initial is not None and initial_simulation is None:
            raise ValueError(
                "initial execution plan requires matching simulator evidence"
            )
        self._initial = (
            None
            if initial is None
            else build_plan_run(
                *initial,
                simulation=cast(SimulationResult, initial_simulation),
                bridge=bridge,
                functions=functions,
                memory_envelopes=initial_memory_envelopes or {},
            )
        )
        self._recurrent = build_plan_run(
            *recurrent,
            simulation=recurrent_simulation,
            bridge=bridge,
            functions=functions,
            memory_envelopes=recurrent_memory_envelopes,
        )
        if (self._initial is None) != (initial_fixed_layout is None):
            raise ValueError(
                "initial execution plan and fixed layout must be provided together"
            )
        self._initial_fixed_layout = initial_fixed_layout
        self._recurrent_fixed_layout = recurrent_fixed_layout
        self.optimizer_state = OptimizerState(
            optimizer,
            state,
            bridge,
            recurrent[0],
            has_initial_plan=self._initial is not None,
            was_lazy=optimizer_state_was_lazy,
            preinitialized=optimizer_state_preinitialized,
        )
        # Materialization uses a short-lived action batch. Replace it with
        # exactly one immutable initial or recurrent plan.
        clear_tasks(self._bridge)
        if self._initial is not None and not self.optimizer_state.initialized:
            assert self._initial_fixed_layout is not None
            self._initial = admit_run(
                self._bridge, self._initial, self._initial_fixed_layout
            )
            self._active_run = self._initial
        else:
            self._recurrent = admit_run(
                self._bridge, self._recurrent, self._recurrent_fixed_layout
            )
            self._active_run = self._recurrent
        self._gradients = {
            state.bridge.objects.alias_for_object(
                item.gradient_object_id
            ): model_parameter
            for item in recurrent[0].gradients
            for model_parameter in (state.model.get_parameter(item.parameter_name),)
        }
        self._invocations = 0
        runs = tuple(run for run in (self._initial, self._recurrent) if run is not None)
        self.timing = ExecutionTiming(
            bridge,
            tuple(
                dict.fromkeys(
                    record.task.task_id for run in runs for record in run.execution
                )
            ),
        )
        self._task_annotations = TaskBoundaryAnnotations(self._bridge)
        self._completion = ReusableCompletionEvent(
            bridge.runtime._runtime_handle, state.device
        )

    @property
    def run_in_force(self) -> _PlanRun:
        """The plan the next invocation runs: the initial plan while a lazy
        optimizer's state does not exist yet, the recurrent plan after."""

        run = (
            self._initial
            if self._initial is not None and not self.optimizer_state.initialized
            else self._recurrent
        )
        if run is None:
            raise AssertionError("initial optimizer plan is unavailable")
        return run

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
        run = self.run_in_force
        if run is not self._active_run:
            clear_tasks(self._bridge)
            if run is self._initial:
                layout = self._initial_fixed_layout
                if layout is None:
                    raise AssertionError("initial fixed layout is unavailable")
                self._initial = admit_run(self._bridge, run, layout)
                run = self._initial
            else:
                self._recurrent = admit_run(
                    self._bridge, run, self._recurrent_fixed_layout
                )
                run = self._recurrent
            self._active_run = run
        if timing is not None:
            self.timing.begin_armed_runtime_trace(timing, step_number)
        self._state.refresh_inputs(inputs)
        return run

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
            self._state.object_store.pop(alias_id, None)

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
