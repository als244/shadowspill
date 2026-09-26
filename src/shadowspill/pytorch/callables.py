"""Public callable objects returned by PyTorch planning."""

from __future__ import annotations

import os
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from shadowspill.diagnostics.timing import InvocationTiming
from shadowspill.planner.diagnostics.plan import PlanReport
from shadowspill.pytorch.diagnostics.step import DiagnosticsHandle, StepResult
from shadowspill.pytorch.execution import ForwardExecutor, TrainingExecutor
from shadowspill.pytorch.guards import InputSignature, validate_training_inputs
from shadowspill.pytorch.invocation import InvocationResult
from shadowspill.pytorch.materialization import (
    MaterializedForwardState,
    TrainingMaterializedState,
)
from shadowspill.pytorch.state.storage import (
    release_plan_owned_state,
    restore_persistent_object_ids,
)
from shadowspill.runtime import Runtime
from shadowspill.runtime.abi import runtime_library
from shadowspill.runtime.plan import (
    adopt_plan,
    release_plan,
    wait_plan_idle,
)
from shadowspill.runtime.residue import reclaim_plan_scoped_residue
from shadowspill.runtime.teardown import prepare_failure_cleanup


def _require_no_plan_sharing_slab(runtime: Runtime, plan_handle: int) -> None:
    """Refuse to close a plan whose slab another open plan's layout lies in.

    The bytes are that plan's too, so it closes first; closing this one would
    take them from under it.
    """

    sharing = sorted(
        int(runtime_library().shadowspill_plan_id(guest))
        for guest, host in runtime._installed.slab_hosts.items()
        if host == plan_handle
    )
    if sharing:
        raise RuntimeError(
            "close the plans whose layouts share this plan's slab first "
            f"(plan ids {sharing})"
        )


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
        adopt_plan(self._runtime, plan_handle)
        self._closed = False
        self._closing = False
        self._trace_prepared = False
        self._profiler_annotations_active = False
        self._pending_diagnostics: DiagnosticsHandle | None = None
        self._pending_invocation: InvocationResult[object] | None = None

    def __call__(
        self,
        inputs: Sequence[Any],
        *,
        runtime_trace: bool = False,
        profiler_annotations: bool = False,
    ) -> object:
        self._require_no_pending_invocation()
        return self._invoke(
            inputs,
            runtime_trace=runtime_trace,
            profiler_annotations=profiler_annotations,
        )

    def submit(
        self,
        inputs: Sequence[Any],
        *,
        runtime_trace: bool = False,
        profiler_annotations: bool = False,
    ) -> InvocationResult[object]:
        """Dispatch without synchronizing and return explicit result ownership."""

        self._require_no_pending_invocation()
        output = self._invoke(
            inputs,
            runtime_trace=runtime_trace,
            profiler_annotations=profiler_annotations,
        )
        try:
            synchronize = self._executor.record_invocation_completion()
        except BaseException as error:
            self._close_after_failure(
                error,
                operation="record planned forward completion",
            )
            raise
        invocation = InvocationResult(
            output,
            synchronize,
            on_resolved=self._finish_pending_invocation,
            on_failure=lambda error: self._close_after_failure(
                error,
                operation="synchronize planned forward",
            ),
        )
        self._pending_invocation = invocation
        return invocation

    def _invoke(
        self,
        inputs: Sequence[Any],
        *,
        runtime_trace: bool,
        profiler_annotations: bool,
    ) -> object:
        if self._closed:
            raise RuntimeError("planned forward callable is closed")
        if (
            self._pending_diagnostics is not None
            and not self._pending_diagnostics.resolved
        ):
            raise RuntimeError(
                "resolve the preceding traced call's diagnostics before "
                "launching another traced call"
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
        prepared_inputs = self._executor.prepare_invocation(inputs)
        self._signature.validate(prepared_inputs)
        # Ownership conflicts are invocation preconditions, not execution
        # failures.  Reject them before entering failure cleanup so releasing
        # the outstanding reference makes this callable immediately reusable.
        self._executor.validate_invocation()
        trace_setup_ns = 0
        if runtime_trace:
            if not self._trace_prepared:
                started_ns = time.perf_counter_ns()
                self._executor.timing.prepare()
                trace_setup_ns = time.perf_counter_ns() - started_ns
                self._trace_prepared = True
            self._executor.arm_runtime_trace(trace_setup_ns=trace_setup_ns)
        try:
            output = self._executor(prepared_inputs)
        except BaseException as error:
            if runtime_trace:
                try:
                    self._executor.timing.cancel()
                except BaseException as timing_error:
                    error.add_note(
                        "Failed to cancel execution timing during fault cleanup: "
                        f"{timing_error}"
                    )
            self._close_after_failure(error, operation="execute planned forward")
            raise
        self._pending_diagnostics = (
            DiagnosticsHandle(self._executor.timing.collect_step_diagnostics)
            if runtime_trace
            else None
        )
        return output

    @property
    def diagnostics(self) -> DiagnosticsHandle | None:
        """The last traced call's diagnostics, or None if it was not traced.

        A training step carries its handle on the `StepResult` it returns. A
        forward call returns the model's own output and nothing else, so its
        handle is here. It resolves the same way, and until it is resolved no
        second traced call may begin.
        """

        return self._pending_diagnostics

    def mark_cycle_end(self) -> None:
        """Close the last invocation's cycle where the next one would begin."""
        self._require_open("mark the cycle's end")
        self._executor.timing.mark_cycle_end()

    def invocation_timings(self) -> tuple[InvocationTiming, ...]:
        """Completed invocations on the device clock, once each, oldest first."""
        self._require_open("read invocation timings")
        return self._executor.timing.invocation_timings()

    def _require_open(self, action: str) -> None:
        if self._closed:
            raise RuntimeError(f"cannot {action} a closed planned forward callable")

    def _require_no_pending_invocation(self) -> None:
        pending = self._pending_invocation
        if pending is not None and not pending.resolved:
            raise RuntimeError(
                "resolve the preceding submitted forward invocation before "
                "reusing this callable"
            )

    def _finish_pending_invocation(
        self,
        invocation: InvocationResult[object],
    ) -> None:
        if self._pending_invocation is invocation:
            self._pending_invocation = None

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
        _require_no_plan_sharing_slab(self._runtime, self._plan_handle)
        self._close(primary_error=None)

    def _close_after_failure(self, error: BaseException, *, operation: str) -> None:
        prepare_failure_cleanup(
            self._runtime,
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
            self._pending_invocation is not None
            and not self._pending_invocation.resolved
        ):
            operations.append(
                (
                    "resolve pending invocation",
                    self._pending_invocation._resolve_for_close,
                )
            )
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
                # The same order the training callable keeps, and for the same
                # reason: giving up an object the runtime still has queued work
                # against is refused.
                (
                    "wait for this plan's work to finish",
                    lambda: wait_plan_idle(self._plan_handle),
                ),
                ("restore model state", self._state.restore_cpu_and_unregister),
                ("release compiled executor", self._release_executor),
                (
                    "reclaim allocations this plan's scopes left behind",
                    lambda: reclaim_plan_scoped_residue(
                        self._runtime, self._plan_handle
                    ),
                ),
                (
                    "release runtime plan",
                    lambda: release_plan(self._runtime, self._plan_handle),
                ),
                (
                    "restore persistent object identities",
                    lambda: restore_persistent_object_ids(self._runtime),
                ),
                (
                    "release plan-owned state",
                    lambda: release_plan_owned_state(self._runtime, self._plan_handle),
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
        wait_plan_idle(self._plan_handle)
        executor = self._executor
        executor.release_timing()
        del self._executor
        del executor

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
        report: PlanReport,
        runtime: Runtime,
        plan_handle: int,
    ) -> None:
        self._model = model
        self._signatures = signatures
        self._executor = executor
        self._state = state
        self.plan_report = report
        self._runtime = runtime
        self._plan_handle = plan_handle
        adopt_plan(self._runtime, plan_handle)
        self._step = 0
        self._closed = False
        self._closing = False
        self._trace_prepared = False
        self._pending_diagnostics: DiagnosticsHandle | None = None
        self._profiler_annotations_active = False
        self._pending_invocation: InvocationResult[StepResult] | None = None

    def __call__(
        self,
        inputs: Sequence[Sequence[Any]],
        *,
        hyperparams: Mapping[str, float | Sequence[float]] | None = None,
        runtime_trace: bool = False,
        profiler_annotations: bool = False,
    ) -> StepResult:
        self._require_no_pending_invocation()
        self._apply_hyperparams(hyperparams)
        return self._invoke(
            inputs,
            runtime_trace=runtime_trace,
            profiler_annotations=profiler_annotations,
        )

    def _apply_hyperparams(
        self, hyperparams: Mapping[str, float | Sequence[float]] | None
    ) -> None:
        """Write this step's hyperparameters into the tensors that carry them.

        A value the program reads from a tensor can change between steps
        without recapturing, because a tensor enters a capture's identity by
        geometry rather than by value. A value read from a plain number was
        fixed when it was captured, so asking to change one is refused rather
        than silently ignored.

        Which values are tensors is settled when the step is planned, by
        ``plan_step``'s own ``hyperparams`` argument naming them. Only the named
        ones are held that way, because an optimizer is entitled to require a
        number, so a name that still holds one is an error naming the fix
        rather than a value silently going nowhere.

        Names resolve against the optimizer's parameter groups and the model's
        named buffers, which are the two registries of named values that
        already exist. Every optimizer group carrying the name is written, so
        one schedule reaches an optimizer with several groups; groups that must
        differ are written directly, which is the mechanism underneath this.
        """

        if not hyperparams:
            return
        groups = self._executor.optimizer_state.optimizer.param_groups
        buffers = dict(self._model.named_buffers())
        for name, value in hyperparams.items():
            in_groups = [
                group[name]
                for group in groups
                if isinstance(group.get(name), torch.Tensor)
            ]
            buffer = buffers.get(name)
            in_buffers = [buffer] if isinstance(buffer, torch.Tensor) else []
            if in_groups and in_buffers:
                raise KeyError(
                    f"{name!r} names both an optimizer value and a model "
                    "buffer, so which one to write is ambiguous; rename one "
                    "or write it directly"
                )
            targets = in_groups or in_buffers
            if targets:
                if not isinstance(value, int | float):
                    raise TypeError(f"{name!r} holds one value, so it takes one number")
                with torch.no_grad():
                    for tensor in targets:
                        tensor.fill_(value)
                continue
            held = [
                item
                for group in groups
                if isinstance(group.get(name), tuple | list)
                for item in group[name]
                if isinstance(item, torch.Tensor)
            ]
            if held:
                # An entry holding several values, as betas does: one number
                # sets them all, a sequence sets them in order.
                if isinstance(value, int | float):
                    values = [float(value)] * len(held)
                else:
                    values = [float(item) for item in value]
                if len(values) != len(held):
                    raise ValueError(
                        f"{name!r} holds {len(held)} values, so it needs that "
                        f"many, not {len(values)}"
                    )
                with torch.no_grad():
                    for tensor, item in zip(held, values, strict=True):
                        tensor.fill_(item)
                continue
            if any(name in group for group in groups) or name in buffers:
                raise TypeError(
                    f"{name!r} is a plain number, so the capture fixed it when "
                    "it was traced. To set it per step, name it when the step "
                    f'is planned -- plan_step(..., hyperparams=("{name}",)) -- '
                    "or register it as a model buffer."
                )
            raise KeyError(f"no optimizer value or model buffer named {name!r}")

    def submit(
        self,
        inputs: Sequence[Sequence[Any]],
        *,
        hyperparams: Mapping[str, float | Sequence[float]] | None = None,
        runtime_trace: bool = False,
        profiler_annotations: bool = False,
    ) -> InvocationResult[StepResult]:
        """Dispatch without synchronizing and return explicit result ownership."""

        self._require_no_pending_invocation()
        self._apply_hyperparams(hyperparams)
        result = self._invoke(
            inputs,
            runtime_trace=runtime_trace,
            profiler_annotations=profiler_annotations,
        )
        try:
            synchronize = self._executor.record_invocation_completion()
        except BaseException as error:
            self._close_after_failure(
                error,
                operation="record planned training completion",
            )
            raise
        invocation = InvocationResult(
            result,
            synchronize,
            on_resolved=self._finish_pending_invocation,
            on_failure=lambda error: self._close_after_failure(
                error,
                operation="synchronize planned training step",
            ),
        )
        self._pending_invocation = invocation
        return invocation

    def _invoke(
        self,
        inputs: Sequence[Sequence[Any]],
        *,
        runtime_trace: bool,
        profiler_annotations: bool,
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
                self._executor.timing.prepare()
                trace_setup_ns = time.perf_counter_ns() - started_ns
                self._trace_prepared = True
            self._executor.timing.arm(
                self._executor.run_in_force.traced_invocation(),
                trace_setup_ns=trace_setup_ns,
            )
        try:
            objectives, metrics = self._executor(inputs, self._step + 1)
        except BaseException as error:
            if runtime_trace:
                try:
                    self._executor.timing.cancel()
                except BaseException as timing_error:
                    error.add_note(
                        "Failed to cancel execution timing during fault cleanup: "
                        f"{timing_error}"
                    )
            self._close_after_failure(error, operation="execute planned training step")
            raise
        self._step += 1
        diagnostics = (
            DiagnosticsHandle(self._executor.timing.collect_step_diagnostics)
            if runtime_trace
            else None
        )
        self._pending_diagnostics = diagnostics
        return StepResult(objectives, metrics, self._step, diagnostics)

    def _require_no_pending_invocation(self) -> None:
        pending = self._pending_invocation
        if pending is not None and not pending.resolved:
            raise RuntimeError(
                "resolve the preceding submitted training invocation before "
                "reusing this callable"
            )

    def _finish_pending_invocation(
        self,
        invocation: InvocationResult[StepResult],
    ) -> None:
        if self._pending_invocation is invocation:
            self._pending_invocation = None

    def mark_cycle_end(self) -> None:
        """Close the last invocation's cycle where the next one would begin."""
        self._require_open("mark the cycle's end")
        self._executor.timing.mark_cycle_end()

    def invocation_timings(self) -> tuple[InvocationTiming, ...]:
        """Completed invocations on the device clock, once each, oldest first."""
        self._require_open("read invocation timings")
        return self._executor.timing.invocation_timings()

    def _collect_prior_invocation_drain_seconds(self) -> float:
        """How long the last call waited for the previous invocation to drain."""

        return self._executor.timing.prior_invocation_drain_seconds

    def state_dict(self) -> dict[str, object]:
        """Synchronously return CPU ``model``, ``optimizer``, and ``step`` state.

        The plan owns the storage holding optimizer state, so the complete
        checkpoint exists only while this callable is open.
        """

        self._require_open("read a checkpoint from")
        return {
            "model": self._state.state_dict(),
            "optimizer": self._executor.optimizer_state.state_dict(),
            "step": self._step,
        }

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write the checkpoint :meth:`state_dict` returns to ``path``, from the pool.

        The state is written from where it is in the spill pool rather than
        copied out of it first: where the pool is one this process can
        address, saving costs no host copy of the state, which on a large
        model is otherwise the largest transient a checkpoint asks for. The
        state stays where it is, and the callable goes on training. Resume
        with ``load_state_dict(torch.load(path, mmap=True))``.

        A model entry the optimizer's state reproduces exactly by a cast -- a
        weight whose master copy the optimizer keeps at another precision,
        say -- is written once, as the optimizer's entry, and the checkpoint
        says which entry it is (``model_from_optimizer``) so that
        :meth:`load_state_dict` can cast it back. Which entry, if any, is
        found from the values at the time of the save, bit for bit, rather
        than assumed from its name or dtype.
        """

        self._require_open("save a checkpoint from")
        with self._executor.optimizer_state.state_dict_in_place() as optimizer:
            model = self._state.state_dict(in_place=True)
            derived = _reproduced_by_optimizer(
                model,
                optimizer,
                _names_by_optimizer_index(
                    self._state.model, self._executor.optimizer_state.optimizer
                ),
            )
            torch.save(
                {
                    "model": OrderedDict(
                        (name, value)
                        for name, value in model.items()
                        if name not in derived
                    ),
                    "optimizer": optimizer,
                    "step": self._step,
                    "model_from_optimizer": derived,
                },
                path,
            )

    def load_state_dict(self, checkpoint: Mapping[str, object]) -> None:
        """Restore a checkpoint :meth:`state_dict` or :meth:`save` produced."""

        self._require_open("restore a checkpoint into")
        if set(checkpoint) - {"model_from_optimizer"} != {"model", "optimizer", "step"}:
            raise RuntimeError("training state_dict keys differ")
        model_state = checkpoint["model"]
        optimizer_state = checkpoint["optimizer"]
        step = checkpoint["step"]
        derived = checkpoint.get("model_from_optimizer", {})
        if (
            not isinstance(model_state, Mapping)
            or not isinstance(optimizer_state, Mapping)
            or not isinstance(derived, Mapping)
        ):
            raise TypeError("training checkpoint model/optimizer must be mappings")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise TypeError("training checkpoint step must be non-negative")
        if derived:
            model_state = _with_reproduced_entries(
                model_state, optimizer_state, derived, self._state.model
            )
        self._state.load_model_state(model_state)
        initialized = self._executor.optimizer_state.load(optimizer_state)
        self._executor.optimizer_state.set_initialized(initialized)
        self._step = step

    def _require_open(self, action: str) -> None:
        if self._closed:
            raise RuntimeError(
                f"cannot {action} a closed planned training callable: optimizer "
                "state was released with the plan; take the checkpoint before "
                "close()"
            )

    def close(self) -> None:
        """Synchronize, give the model back its own storage, release the plan.

        Closing copies nothing, and it moves no weights. ``import_model_state``
        gave this model's parameters storage in the spill pool, and that one
        storage holds the updated weights the whole way through: every step
        both begins and ends with parameters spill-resident, so each step's
        updates are already there. Planning points those same Parameter
        objects at device placeholders for as long as the plan lives; the last
        plan over the model to close points them back. Reading them as
        ordinary CPU tensors is ``export_model_state``, a separate call.

        Optimizer state has no equivalent home. ``plan_step`` builds the
        optimizer with the callable it is given and creates its state in
        storage the plan owns, so unless the caller imported that state there
        is no caller-owned pool for it to be left in. Releasing the plan
        therefore releases the state with it, which is why :meth:`state_dict`
        answers only while this callable is open.
        Resume from a checkpoint with :meth:`load_state_dict`, which writes
        the values into the storage the plan already owns.
        """

        if self._closed:
            return
        _require_no_plan_sharing_slab(self._runtime, self._plan_handle)
        self._close(primary_error=None)

    def _close_after_failure(self, error: BaseException, *, operation: str) -> None:
        prepare_failure_cleanup(
            self._runtime,
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
            self._pending_invocation is not None
            and not self._pending_invocation.resolved
        ):
            operations.append(
                (
                    "resolve pending invocation",
                    self._pending_invocation._resolve_for_close,
                )
            )
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
                # Nothing this plan owns can be given up while the runtime
                # still has work queued against it: unregistering an object a
                # queued action names is refused, and rightly. The wait used
                # to happen inside the executor release, which is after the
                # first thing that gives an object up.
                (
                    "wait for this plan's work to finish",
                    lambda: wait_plan_idle(self._plan_handle),
                ),
                ("clear parameter gradients", self._clear_parameter_gradients),
                ("release optimizer state", self._executor.optimizer_state.release),
                ("restore model state", self._state.restore_cpu_and_unregister),
                ("release compiled executor", self._release_executor),
                (
                    "reclaim allocations this plan's scopes left behind",
                    lambda: reclaim_plan_scoped_residue(
                        self._runtime, self._plan_handle
                    ),
                ),
                (
                    "release runtime plan",
                    lambda: release_plan(self._runtime, self._plan_handle),
                ),
                (
                    "restore persistent object identities",
                    lambda: restore_persistent_object_ids(self._runtime),
                ),
                (
                    "release plan-owned state",
                    lambda: release_plan_owned_state(self._runtime, self._plan_handle),
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
        wait_plan_idle(self._plan_handle)
        executor = self._executor
        executor.release_timing()
        del self._executor
        del executor

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


def _names_by_optimizer_index(
    model: nn.Module, optimizer: torch.optim.Optimizer
) -> dict[int, str]:
    """The model name of each parameter, by its index in an optimizer state_dict."""

    name_of = {id(parameter): name for name, parameter in model.named_parameters()}
    parameters = (
        parameter for group in optimizer.param_groups for parameter in group["params"]
    )
    return {
        index: name_of[id(parameter)]
        for index, parameter in enumerate(parameters)
        if id(parameter) in name_of
    }


def _reproduced_by_optimizer(
    model: Mapping[str, torch.Tensor],
    optimizer: Mapping[str, Any],
    names: Mapping[int, str],
) -> dict[str, tuple[int, str]]:
    """The model entries an optimizer entry reproduces bit for bit by a cast."""

    found: dict[str, tuple[int, str]] = {}
    for index, entries in optimizer["state"].items():
        weight = model.get(names.get(index, ""))
        if weight is None:
            continue
        for key, value in entries.items():
            if (
                isinstance(value, torch.Tensor)
                and value.dtype != weight.dtype
                and value.shape == weight.shape
                and _same_bits(value.to(weight.dtype), weight)
            ):
                found[names[index]] = (index, key)
                break
    return found


def _same_bits(first: torch.Tensor, second: torch.Tensor) -> bool:
    return torch.equal(
        first.contiguous().reshape(-1).view(torch.uint8),
        second.contiguous().reshape(-1).view(torch.uint8),
    )


def _with_reproduced_entries(
    model: Mapping[str, torch.Tensor],
    optimizer: Mapping[str, Any],
    derived: Mapping[str, Any],
    module: nn.Module,
) -> dict[str, torch.Tensor]:
    """A checkpoint's model entries, with those written as optimizer entries
    cast back to the model's dtype."""

    dtypes = {
        name: value.dtype for name, value in module.state_dict(keep_vars=True).items()
    }
    restored = dict(model)
    for name, (index, key) in derived.items():
        restored[name] = optimizer["state"][index][key].to(dtypes[name])
    return restored


__all__ = ["PlannedForward", "PlannedTrainStep"]
