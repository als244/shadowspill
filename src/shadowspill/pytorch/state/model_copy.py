"""Model-structure copying around runtime-owned registered storages."""

from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import cast

import torch
import torch.nn as nn

from .records import PersistentStorage, TensorView


def copy_model_with_runtime_storages(
    model: nn.Module,
    storages: Iterable[PersistentStorage],
    *,
    addressable: bool,
) -> tuple[nn.Module, tuple[PersistentStorage, ...]]:
    """Copy a module hierarchy without copying registered tensor payloads.

    The copy's tensors view the pool the values were just imported into, so
    the payload exists once. That is possible only where this process can
    address the pool; where it cannot, the copy is given host memory of its
    own and the storages stay separate, which means the runtime copies them in
    when a plan adopts them and back out when it is done. Everything else --
    which tensors view which storage, and the deep copy around them -- is the
    same either way, so the two differ in one line.
    """

    memo: dict[int, object] = {}
    imported: list[PersistentStorage] = []
    for storage in storages:
        owner = _runtime_owner(storage) if addressable else _separate_owner(storage)
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
        storage.frontend_storage_is_separate = not addressable
        imported.append(storage)
    copied = copy.deepcopy(model, memo)
    if copied is model:
        raise RuntimeError("model copy unexpectedly retained source identity")
    return copied, tuple(imported)


def _separate_owner(storage: PersistentStorage) -> torch.Tensor:
    """Host memory of the copy's own, holding what was just imported.

    The anchor still names the source model's storage here, and the bytes in
    it are exactly what was written into the pool a moment ago, so a clone is
    both the right values and the right size without reading the pool back.
    """

    return storage.anchor.clone()


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


__all__ = ["copy_model_with_runtime_storages"]
