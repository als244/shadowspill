"""Acquire admitted outputs and hand their allocation ownership to the caller."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from shadowspill.runtime.abi import ObjectBinding
from shadowspill.runtime.failures import RuntimeExecutionError

if TYPE_CHECKING:
    from shadowspill.runtime.plan.bridge import RuntimeBridge


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
        bridge.runtime_library.shadowspill_acquire_objects_handle(
            bridge.runtime._runtime_handle,
            acquisition_handle,
            stream.cuda_stream,
            bindings,
            len(runtime_aliases),
        ),
        "acquire caller outputs",
    )
    expanded = bridge.objects.expand_bindings(alias_ids, runtime_aliases, bindings)
    for alias_id, tensor, binding in zip(alias_ids, tensors, expanded, strict=True):
        if bridge.objects.requires_storage(alias_id):
            torch.ops.shadowspill._acquire_storages([tensor], [binding.pointer])
    return expanded


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
    # The admitted handle excludes CPU/zero-byte roots but retains duplicate
    # device entries. Count every device entry, handing off each root once.
    object_ordinal = 0
    for alias_id, tensor, binding in zip(alias_ids, tensors, bindings, strict=True):
        owns_storage = bridge.objects.requires_storage(alias_id)
        if alias_id not in seen:
            if owns_storage:
                torch.ops.shadowspill._transfer_acquired_storage_to_caller(
                    tensor,
                    acquisition_handle,
                    object_ordinal,
                    binding.generation,
                    binding.allocation_id,
                )
            seen.add(alias_id)
        if owns_storage:
            object_ordinal += 1
