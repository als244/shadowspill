"""Public callable objects returned by PyTorch planning."""

from __future__ import annotations

import os
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, cast

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
from shadowspill.pytorch.state.checkpoint import PoolCheckpoint
from shadowspill.pytorch.state.serialization import (
    decode_tensor_state,
    encode_tensor_state,
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
        int(runtime_library().shadowspill_plan_id(sharer))
        for sharer, owner in runtime._installed.slab_owners.items()
        if owner == plan_handle
    )
    if sharing:
        raise RuntimeError(
            "close the plans whose layouts share this plan's slab first "
            f"(plan ids {sharing})"
        )


def _apply_hyperparams(
    model: nn.Module,
    groups: Sequence[Mapping[str, Any]],
    hyperparams: Mapping[str, float | Sequence[float]] | None,
    write_buffers: Callable[[Mapping[str, torch.Tensor]], None],
) -> None:
    """Write one invocation's hyperparameters into the values that carry them.

    Names resolve against the optimizer's parameter groups -- a step's, since
    a forward has none -- and the model's named buffers, which are the two
    registries of named values that already exist. Every group carrying the
    name is written, so one schedule reaches an optimizer with several
    groups; groups that must differ are written directly, which is the
    mechanism underneath. An entry holding several values, as ``betas``
    does, takes one number for all of them or a sequence in order; a buffer
    takes one number, which fills it.

    Where the value goes differs. An optimizer value the program reads from a
    tensor is a host scalar, written in place: it can change between steps
    without recapturing, because a tensor enters a capture's identity by
    geometry rather than by value. Which values are tensors is settled when
    the step is planned, by ``plan_step``'s own ``hyperparams`` argument
    naming them; a value read from a plain number was fixed when it was
    captured, so asking to change one is refused rather than silently
    ignored, and only the named ones are held, because an optimizer is
    entitled to require a number. A model buffer is a tensor already, and
    state the plan owns: its handle on the module points wherever the
    runtime put the value, so the bytes go through ``write_buffers`` into
    the pool the state lives in.
    """

    if not hyperparams:
        return
    buffers = dict(model.named_buffers())
    registries = "optimizer value or model buffer" if groups else "model buffer"
    written: dict[str, torch.Tensor] = {}
    for name, value in hyperparams.items():
        in_groups = [
            group[name] for group in groups if isinstance(group.get(name), torch.Tensor)
        ]
        buffer = buffers.get(name)
        if in_groups and buffer is not None:
            raise KeyError(
                f"{name!r} names both an optimizer value and a model "
                "buffer, so which one to write is ambiguous; rename one "
                "or write it directly"
            )
        if buffer is not None:
            if not isinstance(value, int | float):
                raise TypeError(f"{name!r} holds one value, so it takes one number")
            written[name] = torch.empty(tuple(buffer.shape), dtype=buffer.dtype).fill_(
                value
            )
            continue
        if in_groups:
            if not isinstance(value, int | float):
                raise TypeError(f"{name!r} holds one value, so it takes one number")
            with torch.no_grad():
                for tensor in in_groups:
                    tensor.fill_(value)
            continue
        tuples = [
            group[name] for group in groups if isinstance(group.get(name), tuple | list)
        ]
        if tuples:
            writes: list[tuple[torch.Tensor, Any]] = []
            for held in tuples:
                if not all(isinstance(item, torch.Tensor) for item in held):
                    raise TypeError(
                        f"{name!r} contains captured constants; declare the key "
                        "in plan_step(..., hyperparams=...) before changing it"
                    )
                values = (
                    [float(value)] * len(held)
                    if isinstance(value, int | float)
                    else [float(item) for item in value]
                )
                if len(values) != len(held):
                    raise ValueError(
                        f"{name!r} holds {len(held)} values per group, so it "
                        f"needs that many, not {len(values)}"
                    )
                writes.extend(zip(held, values, strict=True))
            with torch.no_grad():
                for tensor, item in writes:
                    tensor.fill_(item)
            continue
        if any(name in group for group in groups):
            raise TypeError(
                f"{name!r} is a plain number, so the capture fixed it when "
                "it was traced. To set it per step, name it when the step "
                f'is planned -- plan_step(..., hyperparams=("{name}",)) -- '
                "or register it as a model buffer."
            )
        raise KeyError(f"no {registries} named {name!r}")
    if written:
        write_buffers(written)


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
        hyperparams: Mapping[str, float | Sequence[float]] | None = None,
        runtime_trace: bool = False,
        profiler_annotations: bool = False,
    ) -> object:
        self._require_no_pending_invocation()
        self._apply_hyperparams(hyperparams)
        return self._invoke(
            inputs,
            runtime_trace=runtime_trace,
            profiler_annotations=profiler_annotations,
        )

    def submit(
        self,
        inputs: Sequence[Any],
        *,
        hyperparams: Mapping[str, float | Sequence[float]] | None = None,
        runtime_trace: bool = False,
        profiler_annotations: bool = False,
    ) -> InvocationResult[object]:
        """Dispatch without synchronizing and return explicit result ownership."""

        self._require_no_pending_invocation()
        self._apply_hyperparams(hyperparams)
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

    def _apply_hyperparams(
        self, hyperparams: Mapping[str, float | Sequence[float]] | None
    ) -> None:
        """Write this call's hyperparameters into the model buffers that carry
        them, resolved as :func:`_apply_hyperparams` describes; a forward has
        no optimizer, so a buffer is the one kind of value it can set."""

        _apply_hyperparams(self._model, (), hyperparams, self._write_buffers)

    def _write_buffers(self, values: Mapping[str, torch.Tensor]) -> None:
        if self._closed:
            raise RuntimeError("planned forward callable is closed")
        self._state.write_model_entries(values)

    def prepare_runtime_trace(self) -> None:
        """Allocate reusable trace buffers before starting a measured run."""

        if self._closed:
            raise RuntimeError("planned callable is closed")
        self._require_no_pending_invocation()
        if not self._trace_prepared:
            self._executor.timing.prepare()
            self._trace_prepared = True

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
                self.prepare_runtime_trace()
                trace_setup_ns = time.perf_counter_ns() - started_ns
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

    def synchronize(self) -> None:
        """Return once this callable's work has finished, the end-of-step
        writeback included: a call returns before it has, so that the
        caller's own work between calls overlaps it."""
        self._require_open("wait for")
        wait_plan_idle(self._plan_handle)

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
        from .distributed import current

        prepared = current()
        self._distributed = prepared
        self._distributed_layout = (
            None if prepared is None else prepared.checkpoint_layout()
        )
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
        """Write this step's hyperparameters into the values that carry them:
        the optimizer's parameter groups and the model's buffers, resolved as
        :func:`_apply_hyperparams` describes."""

        if not hyperparams:
            return
        if self._closed:
            raise RuntimeError("planned training callable is closed")
        groups = self._executor.optimizer_state.optimizer.param_groups
        _apply_hyperparams(self._model, groups, hyperparams, self._write_buffers)

    def _write_buffers(self, values: Mapping[str, torch.Tensor]) -> None:
        if self._closed:
            raise RuntimeError("planned training callable is closed")
        self._state.write_model_entries(values)

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

    def prepare_runtime_trace(self) -> None:
        """Allocate reusable trace buffers before starting a measured run."""

        if self._closed:
            raise RuntimeError("planned callable is closed")
        self._require_no_pending_invocation()
        if not self._trace_prepared:
            self._executor.timing.prepare()
            self._trace_prepared = True

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
                self.prepare_runtime_trace()
                trace_setup_ns = time.perf_counter_ns() - started_ns
            self._executor.timing.arm(
                self._executor.run.traced_invocation(),
                trace_setup_ns=trace_setup_ns,
            )
        try:
            objectives, metrics, parameter_metrics = self._executor(
                inputs, self._step + 1
            )
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
        return StepResult(
            objectives, metrics, self._step, diagnostics, parameter_metrics
        )

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

    def synchronize(self) -> None:
        """Return once this callable's work has finished, the end-of-step
        writeback included: a call returns before it has, so that the
        caller's own work between calls overlaps it."""
        self._require_open("wait for")
        wait_plan_idle(self._plan_handle)

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

    def state_dict(
        self, *, weights: Literal["master", "compute"] = "master"
    ) -> dict[str, object]:
        """Synchronously return CPU ``model``, ``optimizer``, and ``step`` state.

        The plan owns the storage holding optimizer state, so the complete
        checkpoint exists only while this callable is open.
        """

        self._require_open("read a checkpoint from")
        optimizer, masters = self._executor.optimizer_state.state_dict()
        return self._checkpoint_payload(
            self._state.state_dict(), optimizer, masters, weights=weights
        )

    def save(
        self,
        path: str | os.PathLike[str],
        *,
        weights: Literal["master", "compute"] = "master",
    ) -> None:
        """Write the checkpoint :meth:`state_dict` returns to ``path``, from the pool.

        Objects are streamed into mapped checkpoint ranges without building
        a full CPU snapshot. Each range is flushed and dropped before the next
        object, including for non-addressable pools. Live state stays in its
        pool, and the callable goes on training. Resume
        with ``load_state_dict(torch.load(path, mmap=True))``. A weight with a
        master copy is written once, as its master by default. With
        ``weights="compute"``, save compute values and upcast them on restore.
        """

        self._require_open("save a checkpoint from")
        writer = PoolCheckpoint(self._state.bridge)
        optimizer, masters = self._executor.optimizer_state.checkpoint_state(writer)
        payload = self._checkpoint_payload(
            self._state.checkpoint_state(writer), optimizer, masters, weights=weights
        )
        payload["model"] = encode_tensor_state(
            cast(Mapping[str, Any], payload["model"])
        )
        writer.save(payload, path)

    def _checkpoint_payload(
        self,
        model: Mapping[str, torch.Tensor],
        optimizer: Mapping[str, object],
        masters: Mapping[str, torch.Tensor],
        *,
        weights: Literal["master", "compute"] = "master",
    ) -> dict[str, object]:
        if weights not in {"master", "compute"}:
            raise ValueError("weights must be 'master' or 'compute'")
        if weights == "compute":
            masters = {}
        if self._distributed_layout is not None:
            names = _weight_names(self._state.model)
            omitted = {alias for name in masters for alias in names[name]}
            return {
                "model": {
                    name: value for name, value in model.items() if name not in omitted
                },
                "optimizer": optimizer,
                "masters": masters,
                "distributed": self._distributed_layout,
                "step": self._step,
            }
        return {
            "model": _with_masters(model, masters, self._state.model),
            "optimizer": optimizer,
            "step": self._step,
        }

    def load_state_dict(self, checkpoint: Mapping[str, object]) -> None:
        """Restore a checkpoint :meth:`state_dict` or :meth:`save` produced."""

        self._require_open("restore a checkpoint into")
        expected_keys = {"model", "optimizer", "step"}
        if self._distributed_layout is not None:
            expected_keys |= {"masters", "distributed"}
            if checkpoint.get("distributed") != self._distributed_layout:
                raise ValueError("checkpoint distributed ownership differs")
        if set(checkpoint) != expected_keys:
            raise RuntimeError("training state_dict keys differ")
        model_state = checkpoint["model"]
        optimizer_state = checkpoint["optimizer"]
        step = checkpoint["step"]
        if not isinstance(model_state, Mapping) or not isinstance(
            optimizer_state, Mapping
        ):
            raise TypeError("training checkpoint model/optimizer must be mappings")
        model_state = decode_tensor_state(
            model_state, self._state.model_state_templates()
        )
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise TypeError("training checkpoint step must be non-negative")
        # A master is where its weights were written; the weights are its cast.
        state = self._executor.optimizer_state
        masters = (
            checkpoint["masters"]
            if self._distributed_layout is not None
            else {
                name: model_state[name]
                for name in state.master_names
                if name in model_state
            }
        )
        if not isinstance(masters, Mapping) or set(masters) - set(state.master_names):
            raise ValueError("checkpoint master parameter inventory differs")
        if self._distributed is not None:
            from .distributed._checkpoint import (
                master_aliases,
                masters_from_compute,
                restore_compute_weights,
            )

            expected = set(self._state.model.state_dict(keep_vars=True))
            omitted = master_aliases(checkpoint, self._distributed)
            if set(model_state) != expected - omitted:
                raise ValueError(
                    "checkpoint model entries differ from non-mastered state"
                )
            weights = self._state.state_dict(in_place=True)
            restore_compute_weights(weights, masters, self._distributed)
            model_state = {**weights, **model_state}
            masters = {
                **masters_from_compute(
                    model_state,
                    set(state.master_names) - set(masters),
                    self._distributed,
                ),
                **masters,
            }
        if set(masters) != set(state.master_names):
            raise ValueError("checkpoint master parameter inventory differs")
        weight_names = _weight_names(self._state.model)
        self._state.load_model_state(
            model_state,
            cast_names=frozenset(
                alias for name in masters for alias in weight_names[name]
            ),
        )
        state.load(optimizer_state, masters)
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


def _weight_names(module: nn.Module) -> dict[str, tuple[str, ...]]:
    """Every name a weight goes by, under the first of them."""

    names: dict[int, list[str]] = {}
    for name, parameter in module.named_parameters(remove_duplicate=False):
        names.setdefault(id(parameter), []).append(name)
    return {every[0]: tuple(every) for every in names.values()}


def _with_masters(
    model: Mapping[str, torch.Tensor],
    masters: Mapping[str, torch.Tensor],
    module: nn.Module,
) -> OrderedDict[str, torch.Tensor]:
    """A model state_dict with each master in place of its weights, under every
    name the weights go by."""

    result = OrderedDict(model)
    if masters:
        names = _weight_names(module)
        for name, value in masters.items():
            for alias in names[name]:
                result[alias] = value
    return result


__all__ = ["PlannedForward", "PlannedTrainStep"]
