"""Run setup operations on state whose authoritative bytes live in a pool.

Addressable pool tensors already work with ordinary PyTorch operations. For
other pools this mode stages only the roots touched by the current operation,
publishes mutations, and returns metadata views instead of retaining payloads.
It is used during initialization and authentic control-value derivation at
plan time, never in compiled training tasks.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextvars import ContextVar
from typing import Any, cast

import torch
import torch.distributed as dist
from torch._higher_order_ops.effects import with_effects, with_effects_dense
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves, tree_map

from shadowspill.pytorch.representations import (
    is_wrapper,
    map_tensor,
    storage_view,
    tensor_components,
)
from shadowspill.runtime import Runtime

from .records import PersistentStorage
from .registry import registry_for
from .storage import _storage_bytes, _write_storage_bytes

_active: ContextVar[pool_values | None] = ContextVar("pool_values", default=None)


@torch.no_grad()
def initialize_storage_(
    destination: torch.Tensor, fill: Callable[[torch.Tensor], object]
) -> torch.Tensor:
    """Fill one physical CPU tensor, reading sharded source data only once.

    ``fill`` must overwrite its supplied scratch tensor completely. During a
    pool initializer, a whole sharded storage calls it only on that storage's
    host-group leader, then distributes owned shards. The callback must not
    contain collectives or rely on running on every rank. This suits checkpoint
    reads; ordinary random initializers can continue using tensor operations.
    Nonsharded tensors and partial views retain ordinary copy semantics. Scratch
    never escapes the call, and a failed sharded fill is reported to every
    group member before any shard is published.
    """
    if destination.device.type != "cpu" or is_wrapper(destination):
        raise ValueError("initialize_storage_ takes a physical CPU tensor")
    mode = _active.get()
    item = (
        None
        if mode is None
        else mode.registry.storage(int(destination.untyped_storage()._cdata))
    )
    whole = (
        item is not None
        and destination.storage_offset() == 0
        and destination.is_contiguous()
        and destination.numel() * destination.element_size() == item.size_bytes
    )
    sharding = item.sharding if whole else None

    def materialize() -> torch.Tensor:
        value = torch.empty(destination.shape, dtype=destination.dtype, device="cpu")
        fill(value)
        if (
            value.shape != destination.shape
            or value.dtype != destination.dtype
            or value.device.type != "cpu"
            or not value.is_contiguous()
        ):
            raise ValueError("storage initializer changed its scratch tensor geometry")
        return value

    if sharding is None:
        destination.copy_(materialize())
        return destination

    leader = dist.get_process_group_ranks(sharding.host_group)[0]
    value = torch.empty(0, dtype=torch.uint8)
    error: Exception | None = None
    status: list[object] = [None]
    if dist.get_rank() == leader:
        try:
            value = materialize().reshape(-1).view(torch.uint8)
        except Exception as caught:
            error = caught
            status[0] = f"{type(caught).__name__}: {caught}"
    dist.broadcast_object_list(status, src=leader, group=sharding.host_group)
    if status[0] is not None:
        raise RuntimeError(f"sharded initialization failed: {status[0]}") from error
    assert mode is not None and item is not None
    _write_storage_bytes(mode.runtime, item, value, synchronize=True)
    return destination

_WRITE_ONLY_INITIALIZERS = frozenset(
    {
        torch.ops.aten.normal_.default,
        torch.ops.aten.uniform_.default,
        torch.ops.aten.zero_.default,
        torch.ops.aten.fill_.Scalar,
    }
)


def _overwrites_storage(
    function: Any,
    value: torch.Tensor,
    size_bytes: int,
    *,
    copy_destination: torch.Tensor | None = None,
    copy_source: torch.Tensor | None = None,
) -> bool:
    """Only whole-root initializers can discard the previous pool contents.

    A partial view still needs the untouched bytes. Restrict this shortcut to
    operators that never read their destination, rather than inferring it from
    a mutation annotation (which also includes read-modify-write operations).
    A whole-root copy also qualifies if its source uses a different storage.
    Copies from an alias must retain the old bytes for the source read.
    """
    return (
        (
            function in _WRITE_ONLY_INITIALIZERS
            or (
                function is torch.ops.aten.copy_.default
                and value is copy_destination
                and copy_source is not None
                and not is_wrapper(copy_source)
                and value.untyped_storage()._cdata
                != copy_source.untyped_storage()._cdata
            )
        )
        and value.storage_offset() == 0
        and value.is_contiguous()
        and value.numel() * value.element_size() == size_bytes
    )


def empty_host_metadata(size_bytes: int) -> torch.Tensor:
    """Allocate no payload except the few bytes used by host control scalars."""
    dispatch = torch.empty(0, dtype=torch.uint8, device="cpu")
    if size_bytes <= 64:
        return dispatch.new_empty(size_bytes)
    return cast(
        torch.Tensor,
        torch.ops.shadowspill._make_unbacked_cpu_storage(dispatch, size_bytes),
    )


def empty_host_tensor(
    shape: Sequence[int],
    *,
    dtype: torch.dtype,
    stride: Sequence[int] | None = None,
) -> torch.Tensor:
    """An uninitialized CPU tensor declaration, with no model-sized allocation."""
    template = (
        torch.empty(shape, dtype=dtype, device="meta")
        if stride is None
        else torch.empty_strided(shape, stride, dtype=dtype, device="meta")
    )
    return storage_view(
        empty_host_metadata(template.untyped_storage().nbytes()),
        dtype,
        template.shape,
        template.stride(),
    )


class pool_values(TorchDispatchMode):  # type: ignore[no-untyped-call]
    """Temporarily make pool values available to setup-time PyTorch operations.

    Peak payload is the union of storage roots used by one operation, plus
    scratch that operation allocates. Initializers should operate on individual
    tensors instead of passing the whole model to a bulk operation.
    """

    def __init__(self, runtime: Runtime) -> None:
        super().__init__()  # type: ignore[no-untyped-call]
        self.runtime = runtime
        self.registry = registry_for(runtime)
        self._scratch: list[torch.Tensor] = []
        self._tokens: list[Any] = []

    def __enter__(self) -> pool_values:
        super().__enter__()
        self._tokens.append(_active.set(self))
        return self

    def __exit__(self, *args: Any) -> None:
        try:
            super().__exit__(*args)  # type: ignore[no-untyped-call]
        finally:
            _active.reset(self._tokens.pop())
            self._scratch.clear()

    def _buffer(self, index: int, size: int) -> torch.Tensor:
        if index == len(self._scratch):
            self._scratch.append(torch.empty(size, dtype=torch.uint8))
        elif self._scratch[index].numel() < size:
            self._scratch[index] = torch.empty(size, dtype=torch.uint8)
        return self._scratch[index][:size]

    def __torch_dispatch__(
        self,
        function: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        kwargs = kwargs or {}
        schema = function._schema
        writes = tuple(
            argument
            for argument in schema.arguments
            if argument.alias_info is not None and argument.alias_info.is_write
        )
        # Read-only views need metadata alone. Avoid fetching a whole tensor
        # just to select a row before initializing it.
        if (
            not writes
            and schema.returns
            and all(result.alias_info is not None for result in schema.returns)
        ):
            return function(*args, **kwargs)

        staged: dict[int, tuple[PersistentStorage, torch.Tensor]] = {}
        originals: dict[tuple[Any, ...], torch.Tensor] = {}
        copy_destination = copy_source = None
        if function is torch.ops.aten.copy_.default:
            copy_destination = args[0] if args else kwargs.get("self")
            copy_source = args[1] if len(args) > 1 else kwargs.get("src")
            if (
                isinstance(copy_destination, torch.Tensor)
                and isinstance(copy_source, torch.Tensor)
                and not is_wrapper(copy_destination)
                and not is_wrapper(copy_source)
                and copy_destination.device.type == copy_source.device.type == "cpu"
                and copy_destination.dtype == copy_source.dtype
                and copy_destination.shape == copy_source.shape
                and copy_source.is_contiguous()
                and not copy_source.is_conj()
                and not copy_source.is_neg()
            ):
                item = self.registry.storage(
                    int(copy_destination.untyped_storage()._cdata)
                )
                source_item = self.registry.storage(
                    int(copy_source.untyped_storage()._cdata)
                )
                if (
                    item is not None
                    and item.unbacked
                    and (source_item is None or not source_item.unbacked)
                    and _overwrites_storage(
                        function,
                        copy_destination,
                        item.size_bytes,
                        copy_destination=copy_destination,
                        copy_source=copy_source,
                    )
                ):
                    # Checkpoint import already owns the complete source bytes.
                    # Publish them directly, without allocating a full-sized
                    # staging copy for an inaccessible destination.
                    _write_storage_bytes(
                        self.runtime,
                        item,
                        copy_source.reshape(-1).view(torch.uint8),
                        synchronize=True,
                    )
                    return copy_destination

        def read(value: Any) -> Any:
            if isinstance(value, torch.Tensor) and is_wrapper(value):
                return map_tensor(value, read)
            if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
                return value
            identity = int(value.untyped_storage()._cdata)
            item = self.registry.storage(identity)
            if item is None or not item.unbacked:
                return value
            if identity not in staged:
                owner = self._buffer(len(staged), item.size_bytes)
                staged[identity] = (
                    item,
                    owner
                    if _overwrites_storage(
                        function,
                        value,
                        item.size_bytes,
                        copy_destination=copy_destination,
                        copy_source=copy_source,
                    )
                    else _storage_bytes(item, runtime=self.runtime, owner=owner),
                )
            owner = staged[identity][1]
            replacement = storage_view(
                owner, value.dtype, value.shape, value.stride(), value.storage_offset()
            )
            originals[_view_key(replacement)] = value
            return replacement

        result = function(*tree_map(read, args), **tree_map(read, kwargs))
        mutated: set[int] = set()
        for index, argument in enumerate(schema.arguments):
            if argument not in writes:
                continue
            value = args[index] if index < len(args) else kwargs.get(argument.name)
            mutated.update(
                int(component.untyped_storage()._cdata)
                for tensor in tree_leaves(value)
                if isinstance(tensor, torch.Tensor)
                for _, component in tensor_components(tensor)
            )
        for identity, (item, owner) in staged.items():
            if identity in mutated:
                _write_storage_bytes(self.runtime, item, owner, synchronize=True)
        by_staging = {
            int(owner.untyped_storage()._cdata): item for item, owner in staged.values()
        }

        def restore(value: Any) -> Any:
            if isinstance(value, torch.Tensor) and is_wrapper(value):
                return map_tensor(value, restore)
            if not isinstance(value, torch.Tensor):
                return value
            item = by_staging.get(int(value.untyped_storage()._cdata))
            if item is None:
                return value
            original = originals.get(_view_key(value))
            if original is not None:
                return original
            return storage_view(
                item.anchor,
                value.dtype,
                value.shape,
                value.stride(),
                value.storage_offset(),
            )

        return tree_map(restore, result)


def _view_key(tensor: torch.Tensor) -> tuple[Any, ...]:
    return (
        int(tensor.untyped_storage()._cdata),
        tensor.dtype,
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.storage_offset(),
    )


# Effect tokens order setup-time operations; the wrapped operation must still
# stage its pool-backed arguments through this mode, just like an ordinary op.
@with_effects.py_impl(pool_values)
def _pool_with_effects(mode: pool_values, *args: Any, **kwargs: Any) -> Any:
    with mode:
        return with_effects_dense(*args, **kwargs)
