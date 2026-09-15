"""The after-task half of one task boundary: outputs classified, gradients
accumulated, storages published to the runtime and to the frontend, the task's
scope closed or aborted.

The compiled result must be dropped before the boundary publishes, so that the
allocator frees of unadopted outputs precede any action reusing their ranges;
`after_task` is written so that no frame above it keeps that result alive.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, cast

import torch
from torch.utils._pytree import tree_flatten

from shadowspill.pytorch.diagnostics.timing import (
    ArmedTaskTiming as _ArmedTaskTiming,
)
from shadowspill.pytorch.runtime_adapter.bridge import (
    PublishedStorage,
    abort_task,
    after_task_and_update,
    describe_object_state,
)
from shadowspill.pytorch.runtime_adapter.failures import (
    RuntimeFailureDiagnostics,
    allocator_oom_error,
    generic_runtime_error,
    read_allocator_failure,
)

from ..records import (
    ExecutionTaskRecord as _ExecutionTaskRecord,
)
from ..records import (
    PlanRun as _PlanRun,
)
from .values import PreparedTask, ProcessedTaskOutputs, same_tensor_view

if TYPE_CHECKING:
    from . import TrainingExecutor


def after_task(
    executor: TrainingExecutor,
    prepared: PreparedTask,
    raw_outputs: object,
) -> tuple[torch.Tensor, ...]:
    """Publish outputs, actions, and cleanup for one frontend task."""

    annotation_id = (
        executor._task_annotations.begin(
            f"shadowspill.after_task.{prepared.record.trace_label}"
        )
        if executor._task_annotations.enabled
        else 0
    )
    try:
        processed, dematerialized = prepare_task_publication(
            executor, prepared, raw_outputs
        )
        # The compiled result tuple owns every unadopted task output.  Drop
        # this outer reference before publishing the task boundary so its
        # allocator frees become causal predecessors of any action that
        # reuses the task's spatial ranges.
        del raw_outputs
        publish_task_to_runtime(executor, prepared, processed, dematerialized)
        publish_frontend_bindings(executor, prepared, processed)
        if (
            prepared.record.released_ephemeral
            or prepared.record.task.task_id == prepared.run.lowered.optimizer_task_id
        ):
            finish_task_cleanup(executor, prepared)
        outputs = processed.outputs
    finally:
        if annotation_id:
            executor._task_annotations.end(annotation_id)
    executor.timing.finish_task(prepared.timing)
    return outputs


def prepare_task_publication(
    executor: TrainingExecutor,
    prepared: PreparedTask,
    raw_outputs: object,
) -> tuple[ProcessedTaskOutputs, tuple[torch.Tensor, ...]]:
    annotation_id = (
        executor._task_annotations.begin(
            f"shadowspill.output_processing.{prepared.record.trace_label}"
        )
        if executor._task_annotations.enabled
        else 0
    )
    try:
        processed = _process_task_outputs(executor, prepared, raw_outputs)
        dematerialized = _dematerialization_tensors(
            executor,
            prepared.record,
            processed.adopted,
        )
    finally:
        if annotation_id:
            executor._task_annotations.end(annotation_id)
    return processed, dematerialized


def publish_task_to_runtime(
    executor: TrainingExecutor,
    prepared: PreparedTask,
    processed: ProcessedTaskOutputs,
    dematerialized: tuple[torch.Tensor, ...],
) -> None:
    annotation_id = (
        executor._task_annotations.begin(
            f"shadowspill.runtime.after_task.{prepared.record.trace_label}"
        )
        if executor._task_annotations.enabled
        else 0
    )
    try:
        _publish_admitted_task(executor, prepared, processed, dematerialized)
    finally:
        if annotation_id:
            executor._task_annotations.end(annotation_id)
    prepared.runtime_scope_open = False


def _publish_admitted_task(
    executor: TrainingExecutor,
    prepared: PreparedTask,
    processed: ProcessedTaskOutputs,
    dematerialized: tuple[torch.Tensor, ...],
) -> None:
    record = prepared.record
    if record.task_handle == 0:
        raise AssertionError("execution task has no admitted handle")
    try:
        after_task_and_update(
            executor._bridge,
            record.task_handle,
            executor._state.device.index or 0,
            processed.adopted,
            tuple(item.publication_ordinal for item in processed.adopted),
            dematerialized,
            replacements=processed.replacements,
        )
    except RuntimeError as error:
        diagnostics = read_allocator_failure(
            executor._bridge.library,
            "after_task storage publication",
            task=record.identity,
        )
        if diagnostics is not None:
            diagnostics = _describe_refusal(executor, diagnostics, record)
            if diagnostics.is_allocator_oom:
                raise allocator_oom_error(diagnostics) from error
            raise generic_runtime_error(diagnostics) from error
        raise RuntimeError(
            "after_task storage publication failed for "
            f"execution_{record.execution_ordinal:06d} "
            f"({record.semantic_name}): {error}"
        ) from error


def publish_frontend_bindings(
    executor: TrainingExecutor,
    prepared: PreparedTask,
    processed: ProcessedTaskOutputs,
) -> None:
    started_ns = time.perf_counter_ns() if prepared.timing is not None else 0
    replacement_by_alias = (
        {item.alias_id: item for item in processed.replacements}
        if processed.replacements
        else {}
    )
    replacement_aliases = (
        processed.replacement_aliases if processed.replacements else ()
    )
    for publication in processed.adopted:
        alias_id = publication.alias_id
        if alias_id in replacement_aliases:
            executor._state.publish_replacement_views(replacement_by_alias[alias_id])
        else:
            executor._state.object_store[alias_id] = publication.tensor
    if processed.optimizer_bindings:
        for object_id, tensor, alias_id in processed.optimizer_bindings:
            executor._state.object_store.setdefault(alias_id, tensor)
            executor._state.object_tensors[object_id] = tensor
        executor.optimizer_state.available = True
    if prepared.timing is not None:
        prepared.timing.dispatch_output_state_publish_ns = (
            time.perf_counter_ns() - started_ns
        )


def finish_task_cleanup(executor: TrainingExecutor, prepared: PreparedTask) -> None:
    started_ns = time.perf_counter_ns() if prepared.timing is not None else 0
    with executor._task_annotations.range(
        f"shadowspill.cleanup.{prepared.record.trace_label}"
    ):
        _cleanup_after_task(executor, prepared)
    if prepared.timing is not None:
        prepared.timing.dispatch_cleanup_ns = time.perf_counter_ns() - started_ns


def _process_task_outputs(
    executor: TrainingExecutor,
    prepared: PreparedTask,
    raw_outputs: object,
) -> ProcessedTaskOutputs:
    outputs: tuple[torch.Tensor, ...] = ()
    adopted: tuple[PublishedStorage, ...] = ()
    replacement_aliases: frozenset[str] = frozenset()
    optimizer_bindings: tuple[tuple[str, torch.Tensor, str], ...] = ()
    entrypoint = prepared.record.entrypoint
    timing = prepared.timing
    if entrypoint.phase == "optimizer":
        started_ns = time.perf_counter_ns() if timing is not None else 0
        if prepared.eager_optimizer and not executor.optimizer_state.available:
            adopted, optimizer_bindings = executor.optimizer_state.created_state(
                prepared.record
            )
        else:
            optimizer_bindings = ()
        if timing is not None:
            timing.dispatch_output_publish_ns = time.perf_counter_ns() - started_ns
    else:
        started_ns = time.perf_counter_ns() if timing is not None else 0
        if isinstance(raw_outputs, (tuple, list)):
            leaves = raw_outputs
        else:
            leaves, _ = tree_flatten(raw_outputs)
        if timing is not None:
            timing.dispatch_output_flatten_ns = time.perf_counter_ns() - started_ns
        started_ns = time.perf_counter_ns() if timing is not None else 0
        if entrypoint.phase == "forward":
            if not all(isinstance(value, torch.Tensor) for value in leaves):
                raise RuntimeError("captured forward graph returned a static leaf")
            tensor_outputs = tuple(cast(torch.Tensor, value) for value in leaves)
            adopted, replacement_aliases = _bind_forward_outputs(
                executor,
                prepared.record,
                tensor_outputs,
                timing,
            )
            if entrypoint.public_output_leaves:
                outputs = tuple(
                    tensor_outputs[index] for index in entrypoint.public_output_leaves
                )
        else:
            adopted = _accumulate_gradients(executor, prepared.record, leaves, timing)
        if timing is not None:
            timing.dispatch_output_publish_ns = time.perf_counter_ns() - started_ns
        del leaves
    replacements = (
        tuple(
            executor._state.replacement_storage_views(alias_id)
            for item in adopted
            for alias_id in (item.alias_id,)
            if alias_id in replacement_aliases
        )
        if replacement_aliases
        else ()
    )
    return ProcessedTaskOutputs(
        outputs,
        adopted,
        replacements,
        optimizer_bindings,
    )


def _cleanup_after_task(executor: TrainingExecutor, prepared: PreparedTask) -> None:
    _forget_released_objects(executor, prepared.run, prepared.record)
    if prepared.record.task.task_id == prepared.run.lowered.optimizer_task_id:
        executor.optimizer_state.initialized = True
        for parameter in executor._gradients.values():
            parameter.grad = None
        for alias_id in executor._gradients:
            executor._state.object_store.pop(alias_id, None)
        for gradient_binding in prepared.run.lowered.gradients:
            executor._state.object_tensors.pop(
                gradient_binding.gradient_object_id, None
            )


def abort_prepared_task(
    executor: TrainingExecutor,
    prepared: PreparedTask,
) -> None:
    if prepared.runtime_scope_open:
        prepared.runtime_scope_open = False
        abort_task(
            executor._bridge,
            prepared.record.task_handle,
        )


def _bind_forward_outputs(
    executor: TrainingExecutor,
    record: _ExecutionTaskRecord,
    outputs: tuple[torch.Tensor, ...],
    timing: _ArmedTaskTiming | None,
) -> tuple[tuple[PublishedStorage, ...], frozenset[str]]:
    started_ns = time.perf_counter_ns() if timing is not None else 0
    adopted: list[PublishedStorage] = []
    replacements: set[str] = set()
    for item in record.forward_outputs:
        tensor = outputs[item.leaf_index]
        if item.adopt and item.publication_ordinal is not None:
            adopted.append(
                PublishedStorage(
                    tensor,
                    item.alias_id,
                    item.publication_ordinal,
                )
            )
        if item.replace:
            replacements.add(item.alias_id)
        else:
            executor._state.object_store.setdefault(item.alias_id, tensor)
            executor._state.object_tensors[item.object_id] = tensor
    if timing is not None:
        timing.dispatch_output_classification_ns = time.perf_counter_ns() - started_ns
    return tuple(adopted), frozenset(replacements)


def _accumulate_gradients(
    executor: TrainingExecutor,
    record: _ExecutionTaskRecord,
    leaves: Sequence[object],
    timing: _ArmedTaskTiming | None,
) -> tuple[PublishedStorage, ...]:
    started_ns = time.perf_counter_ns() if timing is not None else 0
    contributions: list[torch.Tensor] = []
    destinations: list[torch.Tensor] = []
    first: list[tuple[str, str, torch.Tensor, int | None]] = []
    for item in record.gradient_outputs:
        values = [leaves[index] for index in item.leaf_indices]
        if not all(isinstance(value, torch.Tensor) for value in values):
            raise RuntimeError("parameter gradient became non-tensor")
        contribution = cast(torch.Tensor, values[0])
        for additional in values[1:]:
            if not isinstance(additional, torch.Tensor):
                raise AssertionError("validated gradient became non-tensor")
            contribution.add_(additional)
        destination = executor._state.object_store.get(item.alias_id)
        if destination is None:
            first.append(
                (
                    item.object_id,
                    item.alias_id,
                    contribution,
                    item.publication_ordinal,
                )
            )
        elif same_tensor_view(destination, contribution):
            executor._state.object_tensors[item.object_id] = destination
            parameter = executor._gradients.get(item.alias_id)
            if parameter is not None:
                parameter.grad = destination
        else:
            destinations.append(destination)
            contributions.append(contribution)
    if timing is not None:
        timing.dispatch_output_classification_ns = time.perf_counter_ns() - started_ns
    adopted: list[PublishedStorage] = []
    for _object_id, alias_id, contribution, publication_ordinal in first:
        if publication_ordinal is not None:
            adopted.append(
                PublishedStorage(
                    contribution,
                    alias_id,
                    publication_ordinal,
                )
            )
    started_ns = time.perf_counter_ns() if timing is not None else 0
    for object_id, alias_id, contribution, _publication_ordinal in first:
        executor._state.object_store[alias_id] = contribution
        executor._state.object_tensors[object_id] = contribution
        parameter = executor._gradients.get(alias_id)
        if parameter is not None:
            parameter.grad = contribution
    if timing is not None:
        timing.dispatch_output_state_publish_ns = time.perf_counter_ns() - started_ns
    if destinations:
        torch._foreach_add_(destinations, contributions)
    return tuple(adopted)


def _describe_refusal(
    executor: TrainingExecutor,
    diagnostics: RuntimeFailureDiagnostics,
    record: _ExecutionTaskRecord,
) -> RuntimeFailureDiagnostics:
    """Name the action a refusal was about, and the state that refused it.

    The runtime refuses an action because of the state of the object it names, and
    it reports the object. Without the action and the state beside it, the report
    has to be decoded by hand against the plan.
    """

    if diagnostics.object_id is None:
        return diagnostics
    alias_id = executor._bridge.objects.alias_for_runtime_object(diagnostics.object_id)
    refused: str | None = None
    if alias_id is not None:
        for action in record.actions:
            if action.alias_group_id == alias_id:
                refused = (
                    f"{action.kind.name} {alias_id} (trigger {action.trigger_task_id})"
                )
                break
        if refused is None:
            refused = f"no action on {alias_id} at this task"
    return replace(
        diagnostics,
        refused_action=refused,
        object_state=describe_object_state(executor._bridge, diagnostics.object_id),
    )


def _dematerialization_tensors(
    executor: TrainingExecutor,
    record: _ExecutionTaskRecord,
    adopted: tuple[PublishedStorage, ...],
) -> tuple[torch.Tensor, ...]:
    if not record.dematerialize_aliases:
        return ()
    newly_produced = {item.alias_id: item.tensor for item in adopted}
    pending: list[torch.Tensor] = []
    for alias_id in record.dematerialize_aliases:
        tensor = newly_produced.get(alias_id)
        if tensor is None:
            tensor = executor._state.object_store.get(alias_id)
        if tensor is None:
            raise RuntimeError(f"action references unbound object {alias_id!r}")
        pending.append(tensor)
    return tuple(pending)


def _forget_released_objects(
    executor: TrainingExecutor, run: _PlanRun, record: _ExecutionTaskRecord
) -> None:
    del run
    for alias_id, object_ids in record.released_ephemeral:
        executor._state.object_store.pop(alias_id, None)
        for object_id in object_ids:
            executor._state.object_tensors.pop(object_id, None)


__all__ = [
    "abort_prepared_task",
    "after_task",
    "finish_task_cleanup",
    "prepare_task_publication",
    "publish_frontend_bindings",
    "publish_task_to_runtime",
]
