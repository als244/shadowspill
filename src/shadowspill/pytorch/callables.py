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
from shadowspill.pytorch.execution import ForwardExecutor, TrainingExecutor
from shadowspill.pytorch.guards import InputSignature, validate_training_inputs
from shadowspill.pytorch.materialization import (
    MaterializedForwardState,
    TrainingMaterializedState,
)
from shadowspill.pytorch.runtime_adapter import Runtime
from shadowspill.pytorch.state.storage import restore_persistent_object_ids


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
        plan_handle: int,
    ) -> None:
        self._model = model
        self._signature = signature
        self._executor = executor
        self._state = state
        self.plan_report = report
        self._runtime = runtime
        self._plan_handle = plan_handle
        self._runtime._adopt_plan(plan_handle)
        self._closed = False
        self._closing = False
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
        prepared_inputs = self._executor.prepare_invocation(inputs)
        self._signature.validate(prepared_inputs)
        # Ownership conflicts are invocation preconditions, not execution
        # failures.  Reject them before entering failure cleanup so releasing
        # the outstanding reference makes this callable immediately reusable.
        self._executor.validate_invocation()
        try:
            return self._executor(prepared_inputs)
        except BaseException as error:
            self._close_after_failure(
                error, operation="execute planned forward"
            )
            raise

    def state_dict(self) -> OrderedDict[str, torch.Tensor]:
        """Synchronously return a normal CPU model state mapping."""

        if self._closed:
            return OrderedDict(self._model.state_dict())
        return self._state.state_dict()

    def load_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        """Synchronously load a normal model state mapping into the runtime."""

        if self._closed:
            self._model.load_state_dict(state)
            return
        self._state.load_state_dict(state)

    def close(self) -> None:
        """Synchronize, restore the original model to CPU, and release the plan."""

        if self._closed:
            return
        self._close(primary_error=None)

    def _close_after_failure(
        self, error: BaseException, *, operation: str
    ) -> None:
        self._runtime._prepare_failure_cleanup(
            error,
            operation=operation,
            synchronize_unlatched=True,
        )
        self._close(primary_error=error)

    def _close(self, *, primary_error: BaseException | None) -> None:
        if self._closed or self._closing:
            return
        self._closing = True
        operations: list[tuple[str, Any]] = []
        if self._profiler_annotations_active:
            operations.append(
                ("finish profiler annotations", self._finish_profiler_annotations)
            )
        operations.extend(
            (
                ("restore model state", self._state.restore_cpu_and_unregister),
                ("release compiled executor", self._release_executor),
                (
                    "release runtime plan",
                    lambda: self._runtime._release_plan(self._plan_handle),
                ),
                (
                    "restore persistent object identities",
                    lambda: restore_persistent_object_ids(self._runtime),
                ),
            )
        )
        try:
            _run_cleanup_operations(operations, primary_error=primary_error)
        finally:
            self._closed = True
            self._closing = False

    def _finish_profiler_annotations(self) -> None:
        self._executor.finish_profiler_annotations()
        self._profiler_annotations_active = False

    def _release_executor(self) -> None:
        executor = self._executor
        del self._executor
        del executor
        status = int(
            self._runtime._installed.library.shadowspill_pytorch_allocator_wait_idle()
        )
        if status != 0:
            raise RuntimeError(
                f"compiled forward executor did not become idle (status {status})"
            )

    def __enter__(self) -> PlannedForward:
        if self._closed:
            raise RuntimeError("planned forward callable is closed")
        return self

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> None:
        del exception_type, traceback
        if isinstance(exception, BaseException):
            self._close(primary_error=exception)
        else:
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
        plan_handle: int,
    ) -> None:
        self._model = model
        self._signatures = signatures
        self._executor = executor
        self._state = state
        self._optimizer = optimizer
        self.plan_report = report
        self._runtime = runtime
        self._plan_handle = plan_handle
        self._runtime._adopt_plan(plan_handle)
        self._step = 0
        self._closed = False
        self._closing = False
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
        except BaseException as error:
            if runtime_trace:
                try:
                    self._executor.cancel_execution_timing()
                except BaseException as timing_error:
                    error.add_note(
                        "Failed to cancel execution timing during fault cleanup: "
                        f"{timing_error}"
                    )
            self._close_after_failure(
                error, operation="execute planned training step"
            )
            raise
        self._step += 1
        diagnostics = (
            DiagnosticsHandle(self._executor.collect_step_diagnostics)
            if runtime_trace
            else None
        )
        self._pending_diagnostics = diagnostics
        return StepResult(objectives, metrics, self._step, diagnostics)

    def _arm_selected_span_timing(self) -> None:
        """Arm production-like two-event task-span timing."""

        self._executor.arm_selected_span_timing()

    def _collect_selected_span_seconds(self) -> float:
        """Collect production-like two-event task-span timing."""

        return self._executor.collect_selected_span_seconds()

    def state_dict(self) -> dict[str, object]:
        """Synchronously return CPU ``model``, ``optimizer``, and ``step`` state."""

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

    def load_state_dict(self, checkpoint: Mapping[str, object]) -> None:
        """Restore the complete three-key checkpoint produced by ``state_dict``."""

        if set(checkpoint) != {"model", "optimizer", "step"}:
            raise RuntimeError("training state_dict keys differ")
        model_state = checkpoint["model"]
        optimizer_state = checkpoint["optimizer"]
        step = checkpoint["step"]
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
        self._close(primary_error=None)

    def _close_after_failure(
        self, error: BaseException, *, operation: str
    ) -> None:
        self._runtime._prepare_failure_cleanup(
            error,
            operation=operation,
            synchronize_unlatched=True,
        )
        self._close(primary_error=error)

    def _close(self, *, primary_error: BaseException | None) -> None:
        if self._closed or self._closing:
            return
        self._closing = True
        operations: list[tuple[str, Any]] = []
        if (
            self._pending_diagnostics is not None
            and not self._pending_diagnostics.resolved
        ):
            operations.append(
                ("resolve pending diagnostics", self._pending_diagnostics.result)
            )
        if self._profiler_annotations_active:
            operations.append(
                ("finish profiler annotations", self._finish_profiler_annotations)
            )
        operations.extend(
            (
                ("clear parameter gradients", self._clear_parameter_gradients),
                ("restore optimizer state", self._executor.restore_optimizer_cpu),
                ("restore model state", self._state.restore_cpu_and_unregister),
                ("release compiled executor", self._release_executor),
                (
                    "release runtime plan",
                    lambda: self._runtime._release_plan(self._plan_handle),
                ),
                (
                    "restore persistent object identities",
                    lambda: restore_persistent_object_ids(self._runtime),
                ),
            )
        )
        try:
            _run_cleanup_operations(operations, primary_error=primary_error)
        finally:
            self._closed = True
            self._closing = False

    def _finish_profiler_annotations(self) -> None:
        self._executor.finish_profiler_annotations()
        self._profiler_annotations_active = False

    def _release_executor(self) -> None:
        executor = self._executor
        del self._executor
        del executor
        status = int(
            self._runtime._installed.library.shadowspill_pytorch_allocator_wait_idle()
        )
        if status != 0:
            raise RuntimeError(
                f"compiled training executor did not become idle (status {status})"
            )

    def _clear_parameter_gradients(self) -> None:
        for parameter in self._model.parameters():
            parameter.grad = None

    def __enter__(self) -> PlannedTrainStep:
        if self._closed:
            raise RuntimeError("planned training callable is closed")
        return self

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> None:
        del exception_type, traceback
        if isinstance(exception, BaseException):
            self._close(primary_error=exception)
        else:
            self.close()


def _run_cleanup_operations(
    operations: Sequence[tuple[str, Any]],
    *,
    primary_error: BaseException | None,
) -> None:
    """Run every independent teardown operation without masking the cause."""

    failures: list[tuple[str, BaseException]] = []
    for description, operation in operations:
        try:
            operation()
        except BaseException as error:
            failures.append((description, error))
            if primary_error is not None:
                primary_error.add_note(f"Failed to {description}: {error}")
    if primary_error is not None or not failures:
        return
    description, first = failures[0]
    for later_description, later in failures[1:]:
        first.add_note(f"Failed to {later_description}: {later}")
    first.add_note(f"Callable close failed while attempting to {description}")
    raise first


__all__ = ["PlannedForward", "PlannedTrainStep"]
