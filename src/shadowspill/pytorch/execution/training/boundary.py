"""The before-task half of one task boundary: readiness, inputs acquired and
rebound, the call assembled, the compiled task run.

Functions over the executor: they read its bridge, its materialized state, its
timing and its optimizer state, and change nothing but what the runtime and the
timing record.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch

from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.diagnostics.timing import (
    ArmedTaskTiming as _ArmedTaskTiming,
)
from shadowspill.pytorch.optimizer import (
    OpaqueOptimizerArtifact,
)
from shadowspill.pytorch.runtime_adapter.bridge import (
    abort_task,
    before_task_and_acquire,
    input_failure_states,
    wait_task_allocations,
)

from ..records import (
    ExecutionTaskRecord as _ExecutionTaskRecord,
)
from ..records import (
    PlanRun as _PlanRun,
)
from .publication import abort_prepared_task, after_task
from .values import PreparedTask, TaskCall

if TYPE_CHECKING:
    from . import TrainingExecutor


def execute_task(
    executor: TrainingExecutor,
    run: _PlanRun,
    record: _ExecutionTaskRecord,
) -> tuple[torch.Tensor, ...]:
    prepared = before_task(executor, run, record)
    try:
        # Do not retain the compiled result in this caller frame.  The
        # after-task boundary must be able to destroy every unadopted
        # output before it publishes actions that reuse those ranges.
        return after_task(executor, prepared, run_compiled_task(executor, prepared))
    except BaseException:
        abort_prepared_task(executor, prepared)
        executor.timing.finish_task(prepared.timing)
        raise


def before_task(
    executor: TrainingExecutor,
    run: _PlanRun,
    record: _ExecutionTaskRecord,
) -> PreparedTask:
    """Acquire, rebind, and assemble one complete frontend task boundary."""

    timing = executor.timing.begin_task(record.entrypoint)
    # The invocation's timeline needs the stream twice: to mark where its
    # first task's compute starts and where its last task's ends; every
    # other task keeps the default-off fast boundary.
    needs_span_stream = executor.timing.span_pending or (
        record.task.task_id == run.lowered.optimizer_task_id
    )
    if (
        timing is None
        and not executor._task_annotations.enabled
        and not needs_span_stream
    ):
        return _before_task_fast(executor, run, record)
    runtime_scope_open = False
    try:
        with executor._task_annotations.range(
            f"shadowspill.before_task.{record.trace_label}"
        ):
            stream = _resolve_task_stream(timing, needs_span_stream)
            executor.timing.record_task_readiness(timing, stream)
            with executor._task_annotations.range(
                f"shadowspill.storage_rebind.{record.trace_label}"
            ):
                input_tensors = _lookup_task_inputs(executor, record, timing)
                acquire_started_ns = time.perf_counter_ns()
                _acquire_task_inputs(executor, record, stream, input_tensors, timing)
                if timing is not None:
                    timing.dispatch_input_acquire_ns = (
                        time.perf_counter_ns() - acquire_started_ns
                    )
                runtime_scope_open = True
                call = _assemble_task_call(executor, record, timing)
            executor.timing.record_task_inputs_ready(timing, stream)
            with executor._task_annotations.range(
                f"shadowspill.allocation_reuse.{record.trace_label}"
            ):
                # Host time, not the stream's. This call spins until the
                # worker has published every transfer that still owns a
                # range this task will allocate into, so it is the
                # dispatcher waiting, and it is otherwise invisible.
                reuse_started_ns = time.perf_counter_ns()
                wait_task_allocations(
                    executor._bridge,
                    record.task_handle,
                    executor._state.device.index or 0,
                )
                if timing is not None:
                    timing.dispatch_allocation_reuse_ns = (
                        time.perf_counter_ns() - reuse_started_ns
                    )
            prepared = PreparedTask(
                run=run,
                record=record,
                stream=stream,
                arguments=call.arguments,
                function=call.function,
                eager_optimizer=call.eager_optimizer,
                timing=timing,
            )
            executor.timing.record_compute_start(stream)
            executor.timing.record_task_start(timing, stream)
        return prepared
    except BaseException:
        if runtime_scope_open:
            abort_task(executor._bridge, record.task_handle)
        executor.timing.finish_task(timing)
        raise


def _before_task_fast(
    executor: TrainingExecutor,
    run: _PlanRun,
    record: _ExecutionTaskRecord,
) -> PreparedTask:
    """Execute the default-off-observability boundary without cold-path work."""

    runtime_scope_open = False
    try:
        # Resolve the already-admitted storage-only input vector.
        try:
            input_tensors = tuple(
                executor._state.object_store[alias_id]
                for alias_id in record.input_storage_aliases
            )
        except KeyError as error:
            raise RuntimeError(
                f"task input {error.args[0]!r} has no tensor binding"
            ) from error

        # Acquire readiness and install every current storage binding.
        before_task_and_acquire(
            executor._bridge,
            record.task_handle,
            executor._state.device.index or 0,
            input_tensors,
        )
        runtime_scope_open = True

        # Assemble the selected callable and its predecoded arguments.
        call = (
            _assemble_optimizer_call(executor, record)
            if record.entrypoint.phase == "optimizer"
            else _assemble_graph_call(executor, record)
        )
        return PreparedTask(
            run=run,
            record=record,
            stream=None,
            arguments=call.arguments,
            function=call.function,
            eager_optimizer=call.eager_optimizer,
            timing=None,
        )
    except BaseException:
        if runtime_scope_open:
            abort_task(executor._bridge, record.task_handle)
        raise


def _resolve_task_stream(
    timing: _ArmedTaskTiming | None,
    needs_span_stream: bool = False,
) -> torch.cuda.Stream | None:
    needs_python_stream = timing is not None or needs_span_stream
    return torch.cuda.current_stream() if needs_python_stream else None


def _lookup_task_inputs(
    executor: TrainingExecutor,
    record: _ExecutionTaskRecord,
    timing: _ArmedTaskTiming | None,
) -> tuple[torch.Tensor, ...]:
    started_ns = time.perf_counter_ns() if timing is not None else 0
    tensors: list[torch.Tensor] = []
    for alias_id in record.input_storage_aliases:
        tensor = executor._state.object_store.get(alias_id)
        if tensor is None:
            raise RuntimeError(f"task input {alias_id!r} has no tensor binding")
        tensors.append(tensor)
    if timing is not None:
        timing.dispatch_input_lookup_ns = time.perf_counter_ns() - started_ns
    return tuple(tensors)


def _acquire_task_inputs(
    executor: TrainingExecutor,
    record: _ExecutionTaskRecord,
    stream: torch.cuda.Stream | None,
    tensors: tuple[torch.Tensor, ...],
    timing: _ArmedTaskTiming | None,
) -> None:
    try:
        _acquire_input_storages(executor, record, stream, tensors)
    except RuntimeError as error:
        states = input_failure_states(executor._bridge, record.input_aliases)
        detail = "; ".join(states) if states else "all snapshots device-ready"
        raise RuntimeError(f"{error}; input_states=[{detail}]") from error


def _acquire_input_storages(
    executor: TrainingExecutor,
    record: _ExecutionTaskRecord,
    stream: torch.cuda.Stream | None,
    tensors: tuple[torch.Tensor, ...],
) -> None:
    if record.task_handle == 0:
        raise AssertionError("execution task has no admitted handle")
    before_task_and_acquire(
        executor._bridge,
        record.task_handle,
        executor._state.device.index or 0,
        tensors,
    )


def _assemble_task_call(
    executor: TrainingExecutor,
    record: _ExecutionTaskRecord,
    timing: _ArmedTaskTiming | None,
) -> TaskCall:
    started_ns = time.perf_counter_ns() if timing is not None else 0
    if record.entrypoint.phase == "optimizer":
        result = _assemble_optimizer_call(executor, record)
    else:
        result = _assemble_graph_call(executor, record)
    if timing is not None:
        timing.dispatch_argument_assembly_ns = time.perf_counter_ns() - started_ns
    return result


def _assemble_optimizer_call(
    executor: TrainingExecutor, record: _ExecutionTaskRecord
) -> TaskCall:
    artifact = record.entrypoint.artifact
    eager = isinstance(artifact, OpaqueOptimizerArtifact) or (
        not executor.optimizer_state.available
    )
    if eager:
        return TaskCall((), None, True)
    if not isinstance(artifact, GraphArtifact):
        raise RuntimeError("optimizer task has no executable artifact")
    object_ids = record.optimizer_argument_object_ids
    if all(object_id is not None for object_id in object_ids):
        try:
            arguments = tuple(
                executor._state.object_tensors[object_id]
                for object_id in object_ids
                if object_id is not None
            )
        except KeyError as error:
            raise RuntimeError(
                f"optimizer object {error.args[0]!r} is unbound"
            ) from error
    else:
        current = executor.optimizer_state.current_bindings()
        try:
            arguments = tuple(
                current[name].tensor
                for name in record.entrypoint.optimizer_binding_names
            )
        except KeyError as error:
            raise RuntimeError(
                f"optimizer tensor {error.args[0]!r} is unbound"
            ) from error
    return TaskCall(arguments, record.function, False)


def _assemble_graph_call(
    executor: TrainingExecutor, record: _ExecutionTaskRecord
) -> TaskCall:
    if not isinstance(record.entrypoint.artifact, GraphArtifact):
        raise RuntimeError("graph task has no captured artifact")
    if record.argument_template is None:
        raise AssertionError("graph argument template is absent")
    arguments = list(record.argument_template)
    for slot in record.entrypoint.input_slots:
        arguments[slot.leaf_index] = executor._state.object_tensors[slot.object_id]
    return TaskCall(arguments, record.function, False)


def run_compiled_task(executor: TrainingExecutor, prepared: PreparedTask) -> object:
    """Dispatch only the numerical task represented by ``prepared``."""

    if prepared.timing is not None:
        prepared.timing.before_task_exit_ns = time.perf_counter_ns()
    with (
        executor._task_annotations.range(
            f"shadowspill.compiled_call.{prepared.record.trace_label}"
        ),
        torch.no_grad(),
    ):
        if prepared.eager_optimizer:
            raw_outputs: object = executor.optimizer_state.optimizer.step()
        else:
            if prepared.function is None:
                raise AssertionError("compiled task function is unavailable")
            raw_outputs = prepared.function(*prepared.arguments)
    if prepared.timing is not None:
        prepared.timing.after_task_enter_ns = time.perf_counter_ns()
    executor.timing.record_task_end(prepared.timing, prepared.stream)
    if prepared.record.task.task_id == prepared.run.lowered.optimizer_task_id:
        executor.timing.record_compute_end(prepared.stream)
    return raw_outputs


__all__ = ["before_task", "execute_task", "run_compiled_task"]
