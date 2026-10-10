"""Run setup operations on state whose authoritative bytes live in a pool.

Addressable pool tensors already work with ordinary PyTorch operations. For
other pools this mode stages only the roots touched by the current operation,
publishes mutations, and returns metadata views instead of retaining payloads.
It is used during initialization and authentic control-value derivation at
plan time, never in compiled training tasks.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves, tree_map

from shadowspill.pytorch.representations import (
    is_wrapper,
    map_tensor,
    storage_view,
    tensor_components,
)
from shadowspill.runtime import Runtime
from shadowspill.runtime.abi import runtime_library

from .records import PersistentStorage
from .registry import registry_for
from .storage import _require_status, _storage_bytes


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

    def __exit__(self, *args: Any) -> None:
        try:
            super().__exit__(*args)  # type: ignore[no-untyped-call]
        finally:
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
                    _storage_bytes(item, runtime=self.runtime, owner=owner),
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
        for identity in mutated & staged.keys():
            item, owner = staged[identity]
            _require_status(
                runtime_library().shadowspill_write_object(
                    self.runtime._runtime_handle,
                    item.current_object_id,
                    item.pool_id,
                    int(owner.untyped_storage().data_ptr()),
                    item.size_bytes,
                ),
                f"initialize persistent object {item.current_object_id}",
            )
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
