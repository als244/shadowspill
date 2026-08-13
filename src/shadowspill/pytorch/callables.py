"""Public callable objects returned by PyTorch planning."""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from shadowspill.pytorch.diagnostics.plan import PlanReport
from shadowspill.pytorch.diagnostics.step import DiagnosticsHandle, StepResult
from shadowspill.pytorch.execution import (
    ExecutionTiming,
    ForwardExecutor,
    TrainingExecutor,
)
from shadowspill.pytorch.guards import InputSignature, validate_training_inputs
from shadowspill.pytorch.materialization import (
    MaterializedForwardState,
    TrainingMaterializedState,
)
from shadowspill.pytorch.runtime_adapter import Runtime


class PlannedForward:
    """Forward-only callable returned by :func:`plan_forward`.

    The original model is runtime-owned until `close()`. Calls validate the
    complete fixed input signature before writing an input slot or launching a
    task. Returned tensors are ordinary caller-owned allocator records.
    """

    def __init__(
        self,
        model: nn.Module,
        signature: InputSignature,
        executor: ForwardExecutor,
        state: MaterializedForwardState,
        report: PlanReport,
        runtime: Runtime,
    ) -> None:
        self._model = model
        self._signature = signature
        self._executor = executor
        self._state = state
        self.plan_report = report
        self._runtime = runtime
        self._runtime._adopt_plan()
        self._closed = False
        self._profiler_annotations_active = False

    def __call__(
        self,
        inputs: Sequence[Any],
        *,
        profiler_annotations: bool = False,
    ) -> object:
        if self._closed:
            raise RuntimeError("planned forward callable is closed")
        if not isinstance(profiler_annotations, bool):
            raise TypeError("profiler_annotations must be a bool")
        if self._profiler_annotations_active and not profiler_annotations:
            self._executor.finish_profiler_annotations()
            self._profiler_annotations_active = False
        elif profiler_annotations and not self._profiler_annotations_active:
            self._executor.set_profiler_annotations(True)
            self._profiler_annotations_active = True
        self._signature.validate(inputs)
        return self._executor(inputs)

    def state_dict(self) -> OrderedDict[str, torch.Tensor]:
        """Synchronously snapshot model state into ordinary CPU tensors."""

        if self._closed:
            return OrderedDict(self._model.state_dict())
        return self._state.state_dict()

    def load_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        """Synchronously replace the current host-authoritative model state."""

        if self._closed:
            self._model.load_state_dict(state)
            return
        self._state.load_state_dict(state)

    def close(self) -> None:
        """Synchronize, restore the original model to CPU, and release the plan."""

        if self._closed:
            return
        if self._profiler_annotations_active:
            self._executor.finish_profiler_annotations()
            self._profiler_annotations_active = False
        self._state.restore_cpu_and_unregister()
        self._closed = True
        self._runtime._release_plan()

    def __enter__(self) -> PlannedForward:
        if self._closed:
            raise RuntimeError("planned forward callable is closed")
        return self

    def __exit__(self, *exception: object) -> None:
        del exception
        self.close()


class PlannedTrainStep:
    """Accumulated training callable returned by :func:`plan_step`."""

    def __init__(
        self,
        model: nn.Module,
        signatures: tuple[InputSignature, ...],
        executor: TrainingExecutor,
        state: TrainingMaterializedState,
        optimizer: torch.optim.Optimizer,
        report: PlanReport,
        runtime: Runtime,
    ) -> None:
        self._model = model
        self._signatures = signatures
        self._executor = executor
        self._state = state
        self._optimizer = optimizer
        self.plan_report = report
        self._runtime = runtime
        self._runtime._adopt_plan()
        self._step = 0
        self._closed = False
        self._trace_prepared = False
        self._pending_diagnostics: DiagnosticsHandle | None = None
        self._profiler_annotations_active = False

    def __call__(
        self,
        inputs: Sequence[Sequence[Any]],
        *,
        runtime_trace: bool = False,
        profiler_annotations: bool = False,
    ) -> StepResult:
        if self._closed:
            raise RuntimeError("planned training callable is closed")
        if (
            self._pending_diagnostics is not None
            and not self._pending_diagnostics.resolved
        ):
            raise RuntimeError(
                "resolve the preceding traced StepResult diagnostics before "
                "launching another traced step"
            )
        if not isinstance(runtime_trace, bool):
            raise TypeError("runtime_trace must be a bool")
        if not isinstance(profiler_annotations, bool):
            raise TypeError("profiler_annotations must be a bool")
        if self._profiler_annotations_active and not profiler_annotations:
            self._executor.finish_profiler_annotations()
            self._profiler_annotations_active = False
        elif profiler_annotations and not self._profiler_annotations_active:
            self._executor.set_profiler_annotations(True)
            self._profiler_annotations_active = True
        validate_training_inputs(inputs, self._signatures)
        trace_setup_ns = 0
        if runtime_trace:
            if not self._trace_prepared:
                started_ns = time.perf_counter_ns()
                self._executor.prepare_execution_tracing()
                trace_setup_ns = time.perf_counter_ns() - started_ns
                self._trace_prepared = True
            self._executor.arm_compute_timing(trace_setup_ns=trace_setup_ns)
        try:
            objectives, metrics = self._executor(inputs)
        except BaseException:
            if runtime_trace:
                self._executor.cancel_execution_timing()
            raise
        self._step += 1
        diagnostics = (
            DiagnosticsHandle(self._executor.collect_step_diagnostics)
            if runtime_trace
            else None
        )
        self._pending_diagnostics = diagnostics
        return StepResult(objectives, metrics, self._step, diagnostics)

    def _arm_compute_timing(self) -> None:
        """Arm qualification-only first-task-to-final-optimizer timing."""

        self._executor.arm_compute_timing()

    def _collect_compute_seconds(self) -> float:
        """Collect a previously armed qualification timing interval."""

        return self._executor.collect_compute_seconds()

    def _collect_execution_timing(self) -> ExecutionTiming:
        """Collect qualification-only per-task and per-phase timings."""

        return self._executor.collect_execution_timing()

    def _arm_selected_span_timing(self) -> None:
        """Arm production-like two-event task-span timing."""

        self._executor.arm_selected_span_timing()

    def _collect_selected_span_seconds(self) -> float:
        """Collect production-like two-event task-span timing."""

        return self._executor.collect_selected_span_seconds()

    def state_dict(self) -> dict[str, object]:
        """Synchronously snapshot model, optimizer, and logical step state."""

        if self._closed:
            model_state = OrderedDict(self._model.state_dict())
        else:
            model_state = self._state.state_dict()
        return {
            "model": model_state,
            "optimizer": self._optimizer.state_dict()
            if self._closed
            else self._executor.optimizer_state_dict(),
            "step": self._step,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore an exact state produced by :meth:`state_dict`."""

        if set(state) != {"model", "optimizer", "step"}:
            raise RuntimeError("training state_dict keys differ")
        model_state = state["model"]
        optimizer_state = state["optimizer"]
        step = state["step"]
        if not isinstance(model_state, Mapping) or not isinstance(
            optimizer_state, Mapping
        ):
            raise TypeError("training checkpoint model/optimizer must be mappings")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise TypeError("training checkpoint step must be non-negative")
        self._state.load_model_state(model_state)
        initialized = self._executor.load_optimizer_state(optimizer_state)
        self._executor.set_optimizer_state_initialized(initialized)
        self._step = step

    def close(self) -> None:
        if self._closed:
            return
        if (
            self._pending_diagnostics is not None
            and not self._pending_diagnostics.resolved
        ):
            self._pending_diagnostics.result()
        if self._profiler_annotations_active:
            self._executor.finish_profiler_annotations()
            self._profiler_annotations_active = False
        for parameter in self._model.parameters():
            parameter.grad = None
        self._executor.restore_optimizer_cpu()
        self._state.restore_cpu_and_unregister()
        self._closed = True
        self._runtime._release_plan()

    def __enter__(self) -> PlannedTrainStep:
        if self._closed:
            raise RuntimeError("planned training callable is closed")
        return self

    def __exit__(self, *exception: object) -> None:
        del exception
        self.close()


__all__ = ["PlannedForward", "PlannedTrainStep"]
