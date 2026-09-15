"""The step's boundaries on the adapter: what happens while a plan runs.

Publishing the cold materialization, acquiring the caller's outputs, submitting
the initial actions, the before- and after-task boundaries, handing outputs to
the caller and waiting for the plan or the runtime to go idle. The task
boundaries themselves run inside the adapter's storage operators, so the tensors
cross into C once; the rest are the adapter's handle calls.
"""

from __future__ import annotations

import ctypes
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from shadowspill.ir import MemoryAction
from shadowspill.runtime.abi import ObjectBinding
from shadowspill.runtime.failures import RuntimeExecutionError

from .common import PublishedStorage, plan_local_id

if TYPE_CHECKING:
    from shadowspill.pytorch.materialization.replacement import (
        ReplacementStorageViews,
    )

    from . import RuntimeBridge


def publish_initial_tensor(
    bridge: RuntimeBridge, alias_id: str, tensor: torch.Tensor
) -> ObjectBinding:
    """Publish cold materialization through the plan-local object record."""

    if not bridge.objects.requires_storage(alias_id):
        bridge.objects.register_placeholder(alias_id)
        return bridge.objects.zero_binding(alias_id)
    binding = ObjectBinding()
    storage = tensor.untyped_storage()
    bridge.require(
        bridge.runtime_library.shadowspill_plan_publish_initial_allocation(
            bridge.plan_handle,
            plan_local_id(alias_id, "alias_"),
            storage.data_ptr(),
            ctypes.byref(binding),
        ),
        "publish initial plan allocation",
    )
    return binding


def acquire_for_caller(
    bridge: RuntimeBridge,
    alias_ids: tuple[str, ...],
    tensors: tuple[torch.Tensor, ...],
    *,
    acquisition_handle: int,
) -> tuple[ObjectBinding, ...]:
    """Acquire an admitted public-object set without opening a task."""

    if len(alias_ids) != len(tensors):
        raise RuntimeExecutionError("caller output binding count differs")
    runtime_aliases = tuple(
        alias_id for alias_id in alias_ids if bridge.objects.requires_storage(alias_id)
    )
    if not runtime_aliases:
        if acquisition_handle != 0:
            raise RuntimeExecutionError(
                "zero-byte caller outputs must not own an acquisition handle"
            )
        return bridge.objects.expand_bindings(alias_ids, (), ())
    admitted = bridge._admitted_acquisitions.get(runtime_aliases)
    if admitted != acquisition_handle or acquisition_handle == 0:
        raise RuntimeExecutionError(
            "caller output acquisition does not match its admitted object set"
        )
    stream = torch.cuda.current_stream()
    bindings = (ObjectBinding * len(runtime_aliases))()
    bridge.require(
        bridge.library.shadowspill_pytorch_acquire_objects_handle(
            acquisition_handle,
            stream.cuda_stream,
            bindings if runtime_aliases else None,
            len(runtime_aliases),
        ),
        "acquire caller outputs",
    )
    expanded = bridge.objects.expand_bindings(alias_ids, runtime_aliases, bindings)
    for alias_id, tensor, binding in zip(alias_ids, tensors, expanded, strict=True):
        rebind(bridge, tensor, alias_id, binding)
    return expanded


def submit_initial_actions(
    bridge: RuntimeBridge,
    actions: tuple[MemoryAction, ...],
    *,
    task_number: int,
) -> None:
    runtime_actions_values = tuple(
        item for item in actions if bridge.objects.requires_storage(item.alias_group_id)
    )
    if not runtime_actions_values:
        return
    stream = torch.cuda.current_stream()
    admitted = bridge._admitted_action_batches.get(task_number)
    if admitted is None:
        raise RuntimeExecutionError(
            f"initial action batch {task_number} was not admitted"
        )
    handle, expected = admitted
    observed = tuple(
        (action.alias_group_id, action.kind) for action in runtime_actions_values
    )
    if observed != expected:
        raise RuntimeExecutionError(
            "initial action batch changed after admission: "
            f"task={task_number}, expected={expected}, observed={observed}"
        )
    bridge.require(
        bridge.library.shadowspill_pytorch_submit_action_batch_handle(
            handle, stream.cuda_stream
        ),
        "submit admitted initial actions",
    )


def transfer_outputs_to_caller(
    bridge: RuntimeBridge,
    alias_ids: tuple[str, ...],
    tensors: tuple[torch.Tensor, ...],
    bindings: tuple[ObjectBinding, ...],
    *,
    acquisition_handle: int,
) -> None:
    if not (len(alias_ids) == len(tensors) == len(bindings)):
        raise RuntimeExecutionError("caller output lease count differs")
    seen: set[str] = set()
    for object_ordinal, (alias_id, tensor, binding) in enumerate(
        zip(alias_ids, tensors, bindings, strict=True)
    ):
        if alias_id in seen:
            continue
        if not bridge.objects.requires_storage(alias_id):
            bridge.objects.release_zero_generation(alias_id)
            seen.add(alias_id)
            continue
        torch.ops.shadowspill._transfer_acquired_storage_to_caller(
            tensor,
            acquisition_handle,
            object_ordinal,
            binding.generation,
            binding.allocation_id,
        )
        seen.add(alias_id)


def rebind(
    bridge: RuntimeBridge, tensor: torch.Tensor, alias_id: str, binding: ObjectBinding
) -> None:
    if not bridge.objects.requires_storage(alias_id):
        return
    torch.ops.shadowspill._acquire_storages([tensor], [binding.pointer])


def rebind_many(
    bridge: RuntimeBridge,
    items: Sequence[tuple[torch.Tensor, str, ObjectBinding]],
) -> None:
    """Install bindings already validated by the runtime task boundary."""

    materialized = tuple(
        item for item in items if bridge.objects.requires_storage(item[1])
    )
    if not materialized:
        return
    torch.ops.shadowspill._acquire_storages(
        [tensor for tensor, _, _ in materialized],
        [binding.pointer for _, _, binding in materialized],
    )


def before_task_and_acquire(
    bridge: RuntimeBridge,
    task_handle: int,
    device_ordinal: int,
    tensors: Sequence[torch.Tensor],
) -> None:
    """Acquire one predecoded storage-only input vector."""

    torch.ops.shadowspill._before_task_storages(
        tensors,
        task_handle,
        device_ordinal,
    )


def wait_task_allocations(
    bridge: RuntimeBridge,
    task_handle: int,
    device_ordinal: int,
) -> None:
    """Resolve the range-reuse dependencies of this task's allocations."""

    torch.ops.shadowspill._wait_task_allocations(
        task_handle,
        device_ordinal,
    )


def after_task_and_update(
    bridge: RuntimeBridge,
    task_handle: int,
    device_ordinal: int,
    adopted: Sequence[PublishedStorage],
    publication_ordinals: Sequence[int],
    dematerialized: Sequence[torch.Tensor],
    *,
    replacements: Sequence[ReplacementStorageViews] = (),
) -> None:
    """Overwrite logical objects and publish one admitted task boundary."""

    if len(adopted) != len(publication_ordinals):
        raise RuntimeExecutionError(
            "task publication tensors and ordinals have different lengths"
        )
    if not replacements:
        torch.ops.shadowspill._after_task_storages(
            tuple(item.tensor for item in adopted),
            publication_ordinals,
            (),
            (),
            dematerialized,
            task_handle,
            device_ordinal,
        )
        return
    materialized = tuple(enumerate(adopted))
    replacement_by_alias = {item.alias_id: item for item in replacements}
    if len(replacement_by_alias) != len(replacements):
        raise RuntimeExecutionError("task replacement aliases are not unique")
    replacement_aliases = frozenset(replacement_by_alias)
    adopted_aliases = {item.alias_id for item in adopted}
    unknown_replacements = replacement_aliases - adopted_aliases
    if unknown_replacements:
        raise RuntimeExecutionError(
            f"task replacement has no adopted output: {sorted(unknown_replacements)}"
        )
    replacement_tensors: list[torch.Tensor] = []
    replacement_target_indices: list[int] = []
    for target_index, (_index, item) in enumerate(materialized):
        replacement = replacement_by_alias.get(item.alias_id)
        if replacement is None:
            continue
        for tensor in replacement.tensors:
            replacement_tensors.append(tensor)
            replacement_target_indices.append(target_index)
    torch.ops.shadowspill._after_task_storages(
        tuple(item.tensor for _, item in materialized),
        publication_ordinals,
        replacement_tensors,
        replacement_target_indices,
        dematerialized,
        task_handle,
        device_ordinal,
    )


def wait_plan_idle(bridge: RuntimeBridge) -> None:
    """Actively wait only for work owned by this admitted plan."""

    bridge.require(
        bridge.runtime_library.shadowspill_plan_wait_idle(bridge.plan_handle),
        "wait for plan idle",
    )


def wait_idle(bridge: RuntimeBridge) -> None:
    """Wait for runtime-global quiescence at lifecycle boundaries."""

    bridge.require(
        bridge.runtime_library.shadowspill_runtime_wait_idle(
            bridge.runtime._runtime_handle
        ),
        "wait idle",
    )


def abort_task(bridge: RuntimeBridge, task_handle: int) -> None:
    """Close the matching admitted task scope after frontend failure."""

    bridge.require(
        bridge.library.shadowspill_pytorch_abort_task_handle(task_handle),
        "abort admitted task",
    )


__all__ = [
    "abort_task",
    "acquire_for_caller",
    "after_task_and_update",
    "before_task_and_acquire",
    "publish_initial_tensor",
    "rebind",
    "rebind_many",
    "submit_initial_actions",
    "transfer_outputs_to_caller",
    "wait_idle",
    "wait_plan_idle",
    "wait_task_allocations",
]
