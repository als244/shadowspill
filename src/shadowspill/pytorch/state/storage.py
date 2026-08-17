"""Generic tensor-storage import into persistent runtime objects."""

from __future__ import annotations

import ctypes
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch

from shadowspill.pytorch.runtime_adapter.abi import ObjectLocationSnapshot
from shadowspill.pytorch.runtime_adapter.bridge import RuntimeBridge
from shadowspill.pytorch.runtime_adapter.runtime import (
    MemoryPool,
    Runtime,
    RuntimeConfigurationError,
)

from .records import PersistentState, PersistentStorage, TensorView
from .registry import registry_for


@dataclass(frozen=True, slots=True)
class NamedTensor:
    """One diagnostic name and existing tensor identity in public state."""

    name: str
    tensor: torch.Tensor


def import_tensors(
    target: object,
    tensors: Iterable[NamedTensor],
    *,
    runtime: Runtime,
    pool: str,
    release_source: bool,
    _allow_in_progress_plan: bool = False,
) -> PersistentState:
    """Copy unique CPU storages into authoritative runtime-pool objects."""

    _validate_pool(
        runtime,
        pool,
        allow_in_progress_plan=_allow_in_progress_plan,
    )
    registry = registry_for(runtime)
    if registry.get(target) is not None:
        raise RuntimeConfigurationError(
            "state is already imported; export it before importing again"
        )
    created = list(
        register_tensor_storages(
            tensors,
            runtime=runtime,
            pool=pool,
            _allow_in_progress_plan=_allow_in_progress_plan,
        )
    )
    try:
        if release_source and created:
            torch.ops.shadowspill._import_cpu_storages(
                [item.anchor for item in created],
                [item.pool_id for item in created],
                [item.pool_pointer for item in created],
                [item.current_object_id for item in created],
                [item.size_bytes for item in created],
            )
            for item in created:
                item.frontend_storage_is_separate = False
        state = PersistentState(
            target=target,
            pool=pool,
            storages=tuple(created),
            source_owner=None,
        )
        registry.add(
            state,
            allow_in_progress_plan=_allow_in_progress_plan,
        )
        return state
    except BaseException:
        unregister_tensor_storages(created, runtime=runtime)
        raise


def register_tensor_storages(
    tensors: Iterable[NamedTensor],
    *,
    runtime: Runtime,
    pool: str,
    _allow_in_progress_plan: bool = False,
) -> tuple[PersistentStorage, ...]:
    """Copy unique source storages into newly registered runtime objects."""

    selected_pool = _validate_pool(
        runtime,
        pool,
        allow_in_progress_plan=_allow_in_progress_plan,
    )
    roots = _storage_roots(tensors)
    object_ids = runtime._reserve_persistent_object_ids(
        len(roots),
        allow_in_progress_plan=_allow_in_progress_plan,
    )
    library = runtime._installed.library
    created: list[PersistentStorage] = []
    try:
        for object_id, (anchor, views) in zip(object_ids, roots, strict=True):
            size_bytes = int(anchor.untyped_storage().nbytes())
            _require_status(
                library.shadowspill_pytorch_register_object(
                    selected_pool.pool_id,
                    object_id,
                    size_bytes,
                    1,
                    int(anchor.untyped_storage().data_ptr()),
                ),
                f"import persistent object {object_id}",
            )
            snapshot = _snapshot(library, object_id, selected_pool.pool_id)
            pool_pointer = int(snapshot.pointer or 0)
            if not snapshot.has_lease or not snapshot.current:
                raise RuntimeError(
                    f"persistent object {object_id} has no authoritative pool lease"
                )
            created.append(
                PersistentStorage(
                    persistent_object_id=object_id,
                    current_object_id=object_id,
                    pool_id=selected_pool.pool_id,
                    size_bytes=size_bytes,
                    pool_pointer=pool_pointer,
                    anchor=anchor,
                    views=views,
                    frontend_storage_is_separate=True,
                )
            )
        return tuple(created)
    except BaseException:
        unregister_tensor_storages(created, runtime=runtime)
        raise


def own_persistent_state(
    target: object,
    *,
    runtime: Runtime,
    pool: str,
    storages: Iterable[PersistentStorage],
    source_owner: object | None = None,
) -> PersistentState:
    """Publish runtime storage ownership for one frontend object."""

    state = PersistentState(
        target=target,
        pool=pool,
        storages=tuple(storages),
        source_owner=source_owner,
    )
    registry_for(runtime).add(state)
    return state


def unregister_tensor_storages(
    storages: Iterable[PersistentStorage],
    *,
    runtime: Runtime,
) -> None:
    """Release newly registered storages during rollback."""

    library = runtime._installed.library
    for item in reversed(tuple(storages)):
        _require_status(
            library.shadowspill_pytorch_unregister_object(item.current_object_id),
            f"release persistent object {item.current_object_id}",
        )


def release_persistent_tensors(
    target: object,
    *,
    runtime: Runtime,
) -> PersistentState | None:
    """Release runtime objects after their frontend views are no longer used.

    This is an internal ownership primitive.  Unlike ``export_tensors``,
    it performs no copy and no storage rebinding.  The caller must already have
    rebound every live public tensor to independent storage, or be abandoning
    the target during rollback.
    """

    registry = registry_for(runtime)
    state = registry.get(target)
    if state is None:
        return None
    unregister_tensor_storages(state.storages, runtime=runtime)
    return registry.remove(target)


def export_tensors(
    target: object,
    *,
    runtime: Runtime,
    release_runtime: bool,
) -> PersistentState | None:
    """Copy authoritative runtime bytes into ordinary CPU storage roots."""

    runtime._require_state_operation_allowed()
    registry = registry_for(runtime)
    state = registry.get(target)
    if state is None:
        return None
    library = runtime._installed.library
    _require_status(
        library.shadowspill_pytorch_allocator_wait_idle(),
        "wait before state export",
    )
    owners = [
        torch.empty(item.size_bytes, dtype=torch.uint8, device="cpu")
        for item in state.storages
    ]
    for item, owner in zip(state.storages, owners, strict=True):
        _require_status(
            library.shadowspill_pytorch_read_object(
                item.pool_id,
                item.current_object_id,
                item.size_bytes,
                int(owner.untyped_storage().data_ptr()),
            ),
            f"export persistent object {item.current_object_id}",
        )
    if state.storages:
        torch.ops.shadowspill._export_cpu_storages(
            [item.anchor for item in state.storages], owners
        )
    for item in state.storages:
        item.frontend_storage_is_separate = True
    _restore_tensor_views(state)
    if release_runtime:
        for item in state.storages:
            _require_status(
                library.shadowspill_pytorch_unregister_object(item.current_object_id),
                f"release persistent object {item.current_object_id}",
            )
        registry.remove(target)
        return None
    return state


def persistent_state(runtime: Runtime, target: object) -> PersistentState | None:
    """Return internal persistent state for materialization integration."""

    return registry_for(runtime).get(target)


def adopt_persistent_tensor(
    runtime: Runtime,
    target: object,
    tensor: torch.Tensor,
    bridge: RuntimeBridge,
    alias_id: str,
) -> PersistentStorage | None:
    """Adopt the tensor's existing persistent object into one plan alias."""

    state = persistent_state(runtime, target)
    if state is None:
        return None
    item = state.by_storage_identity().get(int(tensor.untyped_storage()._cdata))
    if item is None:
        return None
    library = runtime._installed.library
    if item.frontend_storage_is_separate:
        _require_status(
            library.shadowspill_pytorch_write_object(
                item.pool_id,
                item.current_object_id,
                item.size_bytes,
                int(item.anchor.untyped_storage().data_ptr()),
            ),
            f"refresh persistent object {item.current_object_id}",
        )
    item.current_object_id = bridge.adopt_persistent_object(
        alias_id,
        current_object_id=item.current_object_id,
        pool_id=item.pool_id,
        size_bytes=item.size_bytes,
        pool_pointer=item.pool_pointer,
    )
    return item


def restore_persistent_state(
    runtime: Runtime,
    state: PersistentState | None,
) -> None:
    """Restore frontend tensor views after one adopted plan becomes idle."""

    if state is None:
        return
    library = runtime._installed.library
    for item in state.storages:
        if not item.frontend_storage_is_separate:
            continue
        _require_status(
            library.shadowspill_pytorch_read_object(
                item.pool_id,
                item.current_object_id,
                item.size_bytes,
                int(item.anchor.untyped_storage().data_ptr()),
            ),
            f"restore persistent object {item.current_object_id}",
        )
    _restore_tensor_views(state)


def restore_persistent_object_ids(runtime: Runtime) -> None:
    """Return adopted objects to their stable IDs after task records clear."""

    library = runtime._installed.library
    for state in registry_for(runtime).values():
        for item in state.storages:
            if item.current_object_id == item.persistent_object_id:
                continue
            _require_status(
                library.shadowspill_pytorch_rekey_object(
                    item.current_object_id,
                    item.persistent_object_id,
                ),
                f"restore persistent object ID {item.persistent_object_id}",
            )
            item.current_object_id = item.persistent_object_id


def _storage_roots(
    tensors: Iterable[NamedTensor],
) -> tuple[tuple[torch.Tensor, tuple[TensorView, ...]], ...]:
    grouped: dict[int, tuple[torch.Tensor, list[TensorView]]] = {}
    seen_tensors: set[int] = set()
    for item in tensors:
        tensor = item.tensor
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"state entry {item.name!r} is not a tensor")
        if tensor.device.type != "cpu":
            raise RuntimeConfigurationError(
                f"state entry {item.name!r} must be CPU resident before import"
            )
        if tensor.layout is not torch.strided:
            raise RuntimeConfigurationError(
                f"state entry {item.name!r} must use strided storage"
            )
        storage = tensor.untyped_storage()
        if storage.nbytes() == 0 or id(tensor) in seen_tensors:
            continue
        seen_tensors.add(id(tensor))
        storage_identity = int(storage._cdata)
        entry = grouped.get(storage_identity)
        if entry is None:
            anchor = torch.empty(0, dtype=torch.uint8, device="cpu").set_(storage)
            entry = (anchor, [])
            grouped[storage_identity] = entry
        entry[1].append(
            TensorView(
                tensor=tensor,
                shape=tuple(tensor.shape),
                stride=tuple(tensor.stride()),
                storage_offset=int(tensor.storage_offset()),
                requires_grad=bool(tensor.requires_grad),
            )
        )
    return tuple((anchor, tuple(views)) for anchor, views in grouped.values())


def _restore_tensor_views(state: PersistentState) -> None:
    for storage in state.storages:
        for view in storage.views:
            replacement = torch.empty(0, dtype=view.tensor.dtype, device="cpu").set_(
                storage.anchor.untyped_storage(),
                view.storage_offset,
                view.shape,
                view.stride,
            )
            replacement.requires_grad_(view.requires_grad)
            view.tensor.data = replacement


def _validate_pool(
    runtime: Runtime,
    pool: str,
    *,
    allow_in_progress_plan: bool = False,
) -> MemoryPool:
    runtime._require_state_operation_allowed(
        allow_in_progress_plan=allow_in_progress_plan
    )
    try:
        selected = runtime.pools[pool]
    except KeyError as exc:
        raise RuntimeConfigurationError(f"unknown runtime pool {pool!r}") from exc
    if selected.kind != "pinned_host":
        raise RuntimeConfigurationError(
            "the current PyTorch state import path requires a pinned-host pool"
        )
    return selected


def _snapshot(
    library: Any, object_id: int, pool_id: int
) -> ObjectLocationSnapshot:
    result = ObjectLocationSnapshot()
    _require_status(
        library.shadowspill_pytorch_object_location_snapshot(
            object_id, pool_id, ctypes.byref(result)
        ),
        f"inspect persistent object {object_id}",
    )
    return result


def _require_status(raw_status: Any, operation: str) -> None:
    status = int(raw_status)
    if status != 0:
        raise RuntimeError(f"{operation} failed with status {status}")


__all__ = [
    "NamedTensor",
    "adopt_persistent_tensor",
    "export_tensors",
    "import_tensors",
    "own_persistent_state",
    "persistent_state",
    "register_tensor_storages",
    "release_persistent_tensors",
    "restore_persistent_object_ids",
    "restore_persistent_state",
    "unregister_tensor_storages",
]
