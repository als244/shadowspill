"""Model-structure copying around runtime-owned registered storages."""

from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import cast

import torch
import torch.nn as nn

from .records import PersistentStorage, TensorView


def copy_model_with_spill_storages(
    model: nn.Module,
    storages: Iterable[PersistentStorage],
) -> tuple[nn.Module, tuple[PersistentStorage, ...]]:
    """Copy a module hierarchy without copying registered tensor payloads."""

    memo: dict[int, object] = {}
    relocated: list[PersistentStorage] = []
    for storage in storages:
        owner = _spill_owner(storage)
        views: list[TensorView] = []
        for source_view in storage.views:
            source = source_view.tensor
            replacement = _spill_view(owner, source_view)
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
        storage.source_is_external = False
        relocated.append(storage)
    copied = copy.deepcopy(model, memo)
    if copied is model:
        raise RuntimeError("model copy unexpectedly retained source identity")
    return copied, tuple(relocated)


def _spill_owner(storage: PersistentStorage) -> torch.Tensor:
    dispatch = torch.empty(0, dtype=torch.uint8, device="cpu")
    owner = cast(
        torch.Tensor,
        torch.ops.shadowspill._make_spill_cpu_storage(
            dispatch,
            storage.spill_pointer,
            storage.current_object_id,
            storage.size_bytes,
        ),
    )
    if int(owner.untyped_storage().data_ptr()) != storage.spill_pointer:
        raise RuntimeError("spill-backed CPU storage has the wrong address")
    return owner


def _spill_view(owner: torch.Tensor, view: TensorView) -> torch.Tensor:
    source = view.tensor
    result = torch.empty(0, dtype=source.dtype, device="cpu").set_(
        owner.untyped_storage(),
        view.storage_offset,
        view.shape,
        view.stride,
    )
    if isinstance(source, nn.Parameter):
        parameter = nn.Parameter(result, requires_grad=view.requires_grad)
        parameter.__dict__.update(copy.deepcopy(source.__dict__))
        return parameter
    result.requires_grad_(view.requires_grad)
    return result


__all__ = ["copy_model_with_spill_storages"]
