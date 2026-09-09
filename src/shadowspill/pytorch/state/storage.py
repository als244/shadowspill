"""Generic tensor-storage import into persistent runtime objects."""

from __future__ import annotations

import ctypes
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch.overrides import TorchFunctionMode

from shadowspill.pytorch.runtime_adapter.abi import (
    ObjectLocationSnapshot,
    runtime_library,
)
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
    owning_plan: int | None = None,
    pool_backed: Mapping[int, PersistentStorage] | None = None,
    _allow_in_progress_plan: bool = False,
) -> PersistentState:
    """Copy unique CPU storages into authoritative runtime-pool objects.

    ``owning_plan`` names the plan whose close releases the result. Leave it
    None for state the caller owns, which outlives every plan.
    """

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
            pool_backed=pool_backed,
            _allow_in_progress_plan=_allow_in_progress_plan,
        )
    )
    try:
        # Only a storage that still stands apart from its object is copied
        # into it; one adopted from the pool is already where it belongs.
        separate = [item for item in created if item.frontend_storage_is_separate]
        if release_source and separate:
            torch.ops.shadowspill._import_cpu_storages(
                [item.anchor for item in separate],
                [item.pool_id for item in separate],
                [item.pool_pointer for item in separate],
                [item.current_object_id for item in separate],
                [item.size_bytes for item in separate],
            )
            for item in separate:
                item.frontend_storage_is_separate = False
        state = PersistentState(
            target=target,
            pool=pool,
            storages=tuple(created),
            source_owner=None,
            owning_plan=owning_plan,
        )
        registry.add(
            state,
            allow_in_progress_plan=_allow_in_progress_plan,
        )
        return state
    except BaseException:
        unregister_tensor_storages(created, runtime=runtime)
        raise


def import_state_from_file(
    target: object,
    tensors: Iterable[NamedTensor],
    path: str | os.PathLike[str],
    *,
    runtime: Runtime,
    pool: str,
    owning_plan: int | None = None,
) -> PersistentState:
    """Import a checkpoint's values into ``pool`` without building them first.

    ``torch.load`` ordinarily materializes a whole checkpoint in ordinary host
    memory before an import copies it into the pool, so the peak is the
    checkpoint plus the pool. Mapping the file instead makes the source
    reclaimable page cache, and importing before the copy means the values
    land in pool memory directly: the only anonymous host memory involved is
    whatever the target already occupied.

    The checkpoint must name every tensor in ``tensors`` and agree with each
    on dtype and shape. Raw bytes cannot be converted, so a disagreement is
    refused rather than reinterpreted. Extra names in the file are ignored.
    """

    values = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(values, dict):
        raise RuntimeConfigurationError(
            f"{path} does not hold one mapping of names to tensors; sharded "
            "checkpoints are not supported"
        )
    named = tuple(tensors)
    _require_checkpoint_agrees(named, values, path)
    state = import_tensors(
        target,
        named,
        runtime=runtime,
        pool=pool,
        release_source=True,
        owning_plan=owning_plan,
    )
    # The target's tensors are pool-backed now, so this writes into the pool.
    with torch.no_grad():
        for item in named:
            item.tensor.copy_(values[item.name])
    return state


def _require_checkpoint_agrees(
    tensors: tuple[NamedTensor, ...],
    values: dict[str, Any],
    path: str | os.PathLike[str],
) -> None:
    """Refuse a checkpoint that cannot fill these tensors exactly."""

    for item in tensors:
        stored = values.get(item.name)
        if stored is None:
            raise RuntimeConfigurationError(f"{path} has no entry for {item.name!r}")
        if not isinstance(stored, torch.Tensor):
            raise RuntimeConfigurationError(
                f"{path} entry {item.name!r} is not a tensor"
            )
        if stored.dtype != item.tensor.dtype or stored.shape != item.tensor.shape:
            raise RuntimeConfigurationError(
                f"{path} entry {item.name!r} is {stored.dtype} {tuple(stored.shape)}; "
                f"the target is {item.tensor.dtype} {tuple(item.tensor.shape)}"
            )


def register_tensor_storages(
    tensors: Iterable[NamedTensor],
    *,
    runtime: Runtime,
    pool: str,
    pool_backed: Mapping[int, PersistentStorage] | None = None,
    _allow_in_progress_plan: bool = False,
) -> tuple[PersistentStorage, ...]:
    """Copy unique source storages into newly registered runtime objects.

    ``pool_backed`` names storages that already are pool objects, by storage
    identity: those are adopted as they stand, because planning took them
    from this pool in the first place.
    """

    selected_pool = _validate_pool(
        runtime,
        pool,
        allow_in_progress_plan=_allow_in_progress_plan,
    )
    roots = _storage_roots(tensors)
    # A root whose bytes are already a pool object is adopted where it is:
    # it was taken from the pool to begin with, so importing it would copy
    # the pool into itself under a second name.
    adopted = {
        index: pool_backed[int(anchor.untyped_storage()._cdata)]
        for index, (anchor, _views) in enumerate(roots)
        if pool_backed is not None
        and int(anchor.untyped_storage()._cdata) in pool_backed
    }
    object_ids = runtime._reserve_persistent_object_ids(
        len(roots) - len(adopted),
        allow_in_progress_plan=_allow_in_progress_plan,
    )
    created: list[PersistentStorage] = []
    try:
        pending = iter(object_ids)
        for index, (anchor, views) in enumerate(roots):
            taken = adopted.get(index)
            if taken is not None:
                # Planning took these bytes from this pool, so the object
                # exists: the import only records which tensors view it.
                taken.anchor = anchor
                taken.views = views
                created.append(taken)
                continue
            object_id = next(pending)
            size_bytes = int(anchor.untyped_storage().nbytes())
            _require_status(
                runtime._register_object(
                    object_id,
                    size_bytes,
                    pool_id=selected_pool.pool_id,
                    retain_spill_copy=True,
                    initially_resident=True,
                    source_address=int(anchor.untyped_storage().data_ptr()),
                ),
                f"import persistent object {object_id}",
            )
            snapshot = _snapshot(
                runtime._runtime_handle, object_id, selected_pool.pool_id
            )
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
    owning_plan: int | None = None,
) -> PersistentState:
    """Publish runtime storage ownership for one frontend object."""

    state = PersistentState(
        target=target,
        pool=pool,
        storages=tuple(storages),
        source_owner=source_owner,
        owning_plan=owning_plan,
    )
    registry_for(runtime).add(state)
    return state


def release_plan_owned_state(runtime: Runtime, plan_handle: int) -> tuple[object, ...]:
    """Release every persistent state one plan created, and name the targets.

    State the caller imported carries no owning plan and is left alone, so
    this is the whole of what a closing plan owes: it frees what it made and
    touches nothing it was lent.

    Releasing a lease leaves the frontend tensors that viewed it pointing at
    memory the pool has taken back, so every one of them is emptied first: a
    later use then fails on an empty tensor where it happens, instead of
    reading whatever the pool put there next. Emptying allocates nothing.
    The targets are returned so a caller can say which objects were emptied.
    """

    released: list[object] = []
    for state in registry_for(runtime).values():
        if state.owning_plan != plan_handle:
            continue
        for item in state.storages:
            for tensor in (*(view.tensor for view in item.views), item.anchor):
                tensor.data = torch.empty(0, dtype=tensor.dtype, device=tensor.device)
        release_persistent_tensors(state.target, runtime=runtime)
        released.append(state.target)
    return tuple(released)


def unregister_tensor_storages(
    storages: Iterable[PersistentStorage],
    *,
    runtime: Runtime,
) -> None:
    """Release newly registered storages during rollback."""

    for item in reversed(tuple(storages)):
        _require_status(
            runtime_library().shadowspill_unregister_object(
                runtime._runtime_handle, item.current_object_id
            ),
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
    _require_status(
        runtime_library().shadowspill_runtime_wait_idle(runtime._runtime_handle),
        "wait before state export",
    )
    owners = [
        torch.empty(item.size_bytes, dtype=torch.uint8, device="cpu")
        for item in state.storages
    ]
    for item, owner in zip(state.storages, owners, strict=True):
        _require_status(
            runtime_library().shadowspill_read_object(
                runtime._runtime_handle,
                item.current_object_id,
                item.pool_id,
                int(owner.untyped_storage().data_ptr()),
                item.size_bytes,
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
                runtime_library().shadowspill_unregister_object(
                    runtime._runtime_handle, item.current_object_id
                ),
                f"release persistent object {item.current_object_id}",
            )
        registry.remove(target)
        return None
    return state


def read_state(
    target: object,
    tensors: Iterable[NamedTensor],
    *,
    runtime: Runtime,
    copy: bool = True,
) -> dict[str, torch.Tensor]:
    """Return one target's current values, without rebinding anything.

    ``export_tensors`` rebinds the target's own tensors, which a plan holding
    them cannot survive; this only reads, so it is answerable whenever the
    runtime is idle, plan or no plan. Names come from ``tensors``, so the
    caller decides what its state is called and this stays indifferent to
    what kind of state it is.

    ``copy`` decides where the values live. Copied, they are ordinary host
    memory outside the runtime pools, one buffer per storage root with the
    target's views laid over it, so entries that shared a root still share
    one, and they keep the values they had when this returned. Uncopied, they
    view the pool's own bytes: nothing is allocated, ordinary torch operations
    work, and they must be treated as read-only, because writing through one
    changes runtime state behind the runtime's back. They also stop being
    current the next time the plan runs. A root whose pool copy is not the
    authoritative one is copied either way.
    """

    state = registry_for(runtime).get(target)
    if state is None:
        return {}
    _require_status(
        runtime_library().shadowspill_runtime_wait_idle(runtime._runtime_handle),
        "wait before state read",
    )
    owners: dict[int, torch.Tensor] = {}
    for item in state.storages:
        owners[id(item)] = _storage_bytes(item, runtime=runtime, copy=copy)
    located = {
        id(view.tensor): (item, view) for item in state.storages for view in item.views
    }
    result: dict[str, torch.Tensor] = {}
    for named in tensors:
        found = located.get(id(named.tensor))
        if found is None:
            continue
        item, view = found
        result[named.name] = torch.empty(0, dtype=view.tensor.dtype).set_(
            owners[id(item)].untyped_storage(),
            view.storage_offset,
            view.shape,
            view.stride,
        )
    return result


def _storage_bytes(
    item: PersistentStorage,
    *,
    runtime: Runtime,
    copy: bool,
) -> torch.Tensor:
    """Return one root's bytes, copied out of the pool or viewed in it."""

    if not copy:
        snapshot = _snapshot(
            runtime._runtime_handle, item.current_object_id, item.pool_id
        )
        if snapshot.current and snapshot.pointer:
            window = (ctypes.c_uint8 * item.size_bytes).from_address(
                int(snapshot.pointer)
            )
            return torch.frombuffer(window, dtype=torch.uint8)
    owner = torch.empty(item.size_bytes, dtype=torch.uint8, device="cpu")
    _require_status(
        runtime_library().shadowspill_read_object(
            runtime._runtime_handle,
            item.current_object_id,
            item.pool_id,
            int(owner.untyped_storage().data_ptr()),
            item.size_bytes,
        ),
        f"read persistent object {item.current_object_id}",
    )
    return owner


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
    if item.frontend_storage_is_separate:
        _require_status(
            runtime_library().shadowspill_write_object(
                runtime._runtime_handle,
                item.current_object_id,
                item.pool_id,
                int(item.anchor.untyped_storage().data_ptr()),
                item.size_bytes,
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
    for item in state.storages:
        if not item.frontend_storage_is_separate:
            continue
        _require_status(
            runtime_library().shadowspill_read_object(
                runtime._runtime_handle,
                item.current_object_id,
                item.pool_id,
                int(item.anchor.untyped_storage().data_ptr()),
                item.size_bytes,
            ),
            f"restore persistent object {item.current_object_id}",
        )
    _restore_tensor_views(state)


def restore_persistent_object_ids(runtime: Runtime) -> None:
    """Return adopted objects to their stable IDs after task records clear."""

    for state in registry_for(runtime).values():
        for item in state.storages:
            if item.current_object_id == item.persistent_object_id:
                continue
            _require_status(
                runtime_library().shadowspill_rekey_object(
                    runtime._runtime_handle,
                    item.current_object_id,
                    item.persistent_object_id,
                ),
                f"restore persistent object ID {item.persistent_object_id}",
            )
            item.current_object_id = item.persistent_object_id


# --- planning memory --------------------------------------------------------
#: Below this, planning takes host memory the ordinary way: a pool object
#: costs more bookkeeping than the memory it saves, and planning makes many
#: small temporaries.
PLANNING_MEMORY_MINIMUM_BYTES = 1 << 20



def take_pool_memory(
    runtime: Runtime,
    pool: MemoryPool,
    *,
    size_bytes: int,
) -> PersistentStorage:
    """Take ``size_bytes`` from ``pool`` as host memory the caller may write.

    The object is registered without a source, so nothing is copied: the
    caller writes the values it wants where they will stay. The result is an
    ordinary persistent storage with no views yet, which is what an import
    fills in when this memory becomes some object's state.
    """

    if size_bytes <= 0:
        raise ValueError("planning memory needs a positive size")
    object_id = runtime._reserve_persistent_object_ids(
        1, allow_in_progress_plan=True
    )[0]
    _require_status(
        runtime._register_object(
            object_id,
            size_bytes,
            pool_id=pool.pool_id,
            retain_spill_copy=True,
            initially_resident=True,
        ),
        f"take {size_bytes} bytes of planning memory from pool {pool.name!r}",
    )
    pointer = pool_object_pointer(runtime, object_id, pool.pool_id)
    dispatch = torch.empty(0, dtype=torch.uint8, device="cpu")
    anchor = cast(
        torch.Tensor,
        torch.ops.shadowspill._make_runtime_cpu_storage(
            dispatch, pool.pool_id, pointer, object_id, size_bytes
        ),
    )
    if int(anchor.untyped_storage().data_ptr()) != pointer:
        raise RuntimeError("planning memory from the pool has the wrong address")
    return PersistentStorage(
        persistent_object_id=object_id,
        current_object_id=object_id,
        pool_id=pool.pool_id,
        size_bytes=size_bytes,
        pool_pointer=pointer,
        anchor=anchor,
        views=(),
        frontend_storage_is_separate=False,
    )


class PoolTensorFactory(TorchFunctionMode):
    """Serve host tensor factories from the spill pool while this is active.

    Only ordinary host tensors are served, and only above ``minimum_bytes``:
    a fake or meta tensor costs nothing to begin with, and a small one is not
    worth an object. Anything else falls through to PyTorch untouched, so a
    call this does not understand behaves exactly as it always did.
    """

    #: Factories whose result this can produce directly, with how to fill it.
    _FILLED_FROM_SHAPE = ("zeros", "ones", "empty", "full")
    _FILLED_FROM_TENSOR = ("zeros_like", "ones_like", "empty_like", "full_like")

    def __init__(
        self,
        runtime: Runtime,
        pool: MemoryPool,
        *,
        minimum_bytes: int = PLANNING_MEMORY_MINIMUM_BYTES,
    ) -> None:
        super().__init__()
        self._runtime = runtime
        self._pool = pool
        self._minimum_bytes = minimum_bytes
        self._allocations: dict[int, PersistentStorage] = {}

    @property
    def allocations(self) -> dict[int, PersistentStorage]:
        """Every allocation still held, by the identity of its storage."""

        return dict(self._allocations)

    def release_all_but(self, kept: frozenset[int]) -> None:
        """Give back every allocation whose storage is not in ``kept``."""

        for identity, allocation in tuple(self._allocations.items()):
            if identity in kept:
                continue
            del self._allocations[identity]
            allocation.anchor = torch.empty(0, dtype=torch.uint8, device="cpu")
            unregister_tensor_storages((allocation,), runtime=self._runtime)

    def __torch_function__(
        self,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        kwargs = kwargs or {}
        served = self._serve(func, args, kwargs)
        if served is not None:
            return served
        return func(*args, **kwargs)

    def _serve(
        self,
        func: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> torch.Tensor | None:
        name = getattr(func, "__name__", "")
        if name in self._FILLED_FROM_TENSOR:
            return self._like(name, args, kwargs)
        if name in self._FILLED_FROM_SHAPE:
            return self._shaped(name, args, kwargs)
        return None

    def _like(
        self,
        name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> torch.Tensor | None:
        if not args or not isinstance(args[0], torch.Tensor):
            return None
        source = args[0]
        dtype = kwargs.get("dtype") or source.dtype
        device = torch.device(kwargs.get("device") or source.device)
        result = self._tensor(tuple(source.shape), dtype, device, kwargs)
        if result is None:
            return None
        if name == "zeros_like":
            return result.zero_()
        if name == "ones_like":
            return result.fill_(1)
        if name == "full_like":
            value = kwargs.get("fill_value", args[1] if len(args) > 1 else 0)
            return result.fill_(value)
        return result

    def _shaped(
        self,
        name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> torch.Tensor | None:
        shape = _requested_shape(name, args)
        if shape is None:
            return None
        dtype = kwargs.get("dtype") or torch.get_default_dtype()
        device = torch.device(kwargs.get("device") or "cpu")
        result = self._tensor(shape, dtype, device, kwargs)
        if result is None:
            return None
        if name == "zeros":
            return result.zero_()
        if name == "ones":
            return result.fill_(1)
        if name == "full":
            value = kwargs.get("fill_value", args[1] if len(args) > 1 else 0)
            return result.fill_(value)
        return result

    def _tensor(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        kwargs: dict[str, Any],
    ) -> torch.Tensor | None:
        if device.type != "cpu" or kwargs.get("out") is not None:
            return None
        if kwargs.get("layout") not in (None, torch.strided):
            return None
        if kwargs.get("pin_memory"):
            # Pool memory is pinned already, but saying so is the caller's
            # contract with PyTorch, not ours to reinterpret.
            return None
        elements = 1
        for size in shape:
            if size < 0:
                return None
            elements *= size
        size_bytes = elements * torch.empty(0, dtype=dtype).element_size()
        if size_bytes < self._minimum_bytes:
            return None
        # PyTorch pops this mode while `__torch_function__` runs, so the
        # calls below are ordinary allocations, not another visit here.
        allocation = take_pool_memory(
            self._runtime, self._pool, size_bytes=size_bytes
        )
        view = torch.empty(0, dtype=dtype, device="cpu")
        view.set_(
            allocation.anchor.untyped_storage(),
            0,
            torch.Size(shape),
            _contiguous_stride(shape),
        )
        self._allocations[int(view.untyped_storage()._cdata)] = allocation
        if kwargs.get("requires_grad"):
            view.requires_grad_(True)
        return view


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride: list[int] = []
    running = 1
    for size in reversed(shape):
        stride.append(running)
        running *= size
    return tuple(reversed(stride))


def _requested_shape(name: str, args: tuple[Any, ...]) -> tuple[int, ...] | None:
    """The shape a size-taking factory was asked for, or None if it is unclear."""

    if not args:
        return None
    first = args[0]
    if isinstance(first, torch.Size | list | tuple):
        values = tuple(first)
    else:
        values = tuple(
            item for item in (args[:-1] if name == "full" else args) if item is not None
        )
    if not values or any(not isinstance(item, int) for item in values):
        return None
    return tuple(int(item) for item in values)


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


def pool_object_pointer(runtime: Runtime, object_id: int, pool_id: int) -> int:
    """Where one registered object's bytes live in its pool."""

    snapshot = _snapshot(runtime._runtime_handle, object_id, pool_id)
    if not snapshot.has_lease or not snapshot.current:
        raise RuntimeError(f"persistent object {object_id} has no authoritative lease")
    return int(snapshot.pointer or 0)


def unregister_pool_object(runtime: Runtime, object_id: int) -> None:
    """Give one registered object back to its pool."""

    _require_status(
        runtime_library().shadowspill_unregister_object(
            runtime._runtime_handle, object_id
        ),
        f"release persistent object {object_id}",
    )


def _snapshot(
    runtime_handle: int, object_id: int, pool_id: int
) -> ObjectLocationSnapshot:
    result = ObjectLocationSnapshot()
    _require_status(
        runtime_library().shadowspill_object_location_snapshot(
            runtime_handle, object_id, pool_id, ctypes.byref(result)
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
