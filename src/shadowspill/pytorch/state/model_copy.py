"""Model-structure copying around runtime-owned registered storages."""

from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import cast

import torch
import torch.nn as nn

from shadowspill.pytorch.distributed import borrowed_group_memo
from shadowspill.pytorch.representations import map_tensor, storage_view

from .records import PersistentStorage, TensorView


def copy_model_with_runtime_storages(
    model: nn.Module,
    storages: Iterable[PersistentStorage],
    *,
    addressable: bool,
) -> tuple[nn.Module, tuple[PersistentStorage, ...]]:
    """Copy a module hierarchy without copying registered tensor payloads.

    Addressable pools supply CPU views of their leases. Other pools supply
    guarded metadata-only owners; explicit reads fetch values on demand.
    Both preserve ties and views without retaining a second host payload.
    """

    memo: dict[int, object] = borrowed_group_memo()
    imported: list[PersistentStorage] = []
    for storage in storages:
        owner = _runtime_owner(storage) if addressable else _unbacked_owner(storage)
        views: list[TensorView] = []
        for source_view in storage.views:
            source = source_view.tensor
            replacement = _runtime_view(owner, source_view)
            memo[id(source)] = replacement
            views.append(
                TensorView(
                    tensor=replacement,
                    shape=source_view.shape,
                    stride=source_view.stride,
                    storage_offset=source_view.storage_offset,
                    requires_grad=source_view.requires_grad,
                )
            )
        storage.anchor = owner
        storage.views = tuple(views)
        storage.frontend_storage_is_separate = False
        storage.unbacked = not addressable
        imported.append(storage)

    def empty_leaf(value: torch.Tensor) -> torch.Tensor:
        if value.untyped_storage().nbytes():
            raise RuntimeError("imported model component has no storage registration")
        return copy.deepcopy(value, memo)

    for _, tensor in (
        *model.named_parameters(remove_duplicate=False),
        *model.named_buffers(remove_duplicate=False),
    ):
        # Nonempty leaves are already in memo; empty state has no pool storage.
        map_tensor(tensor, empty_leaf, memo=memo)
    copied = copy.deepcopy(model, memo)
    if copied is model:
        raise RuntimeError("model copy unexpectedly retained source identity")
    return copied, tuple(imported)


def _unbacked_owner(storage: PersistentStorage) -> torch.Tensor:
    return cast(
        torch.Tensor,
        torch.ops.shadowspill._make_unbacked_cpu_storage(
            torch.empty(0, dtype=torch.uint8), storage.size_bytes
        ),
    )


def _runtime_owner(storage: PersistentStorage) -> torch.Tensor:
    dispatch = torch.empty(0, dtype=torch.uint8, device="cpu")
    owner = cast(
        torch.Tensor,
        torch.ops.shadowspill._make_runtime_cpu_storage(
            dispatch,
            storage.pool_id,
            storage.pool_pointer,
            storage.current_object_id,
            storage.size_bytes,
        ),
    )
    if int(owner.untyped_storage().data_ptr()) != storage.pool_pointer:
        raise RuntimeError("runtime-backed CPU storage has the wrong address")
    return owner


def _runtime_view(owner: torch.Tensor, view: TensorView) -> torch.Tensor:
    source = view.tensor
    result = storage_view(
        owner, source.dtype, view.shape, view.stride, view.storage_offset
    )
    if isinstance(source, nn.Parameter):
        parameter = nn.Parameter(result, requires_grad=view.requires_grad)
        parameter.__dict__.update(copy.deepcopy(source.__dict__, borrowed_group_memo()))
        return parameter
    result.requires_grad_(view.requires_grad)
    return result


__all__ = ["copy_model_with_runtime_storages"]
