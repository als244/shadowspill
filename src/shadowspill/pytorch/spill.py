"""A CPU tensor's bytes, as the runtime takes them: an address and a size.

The runtime's spill objects are byte ranges. A frontend that keeps its state in
tensors has to say which bytes, and PyTorch answers with a storage pointer and a
length -- which is the whole of what `PlanObjects` ever used a tensor for. These
three are that conversion, and the residency check that goes with it: the address
handed over must be host memory, and only the framework can say whether it is.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from shadowspill.errors import PlanningError
from shadowspill.runtime.failures import RuntimeExecutionError

if TYPE_CHECKING:
    from shadowspill.pytorch.state.records import PersistentStorage
    from shadowspill.runtime.plan.objects import PlanObjects


def _host_bytes(
    tensor: torch.Tensor, whose: str, error: type[Exception]
) -> tuple[int, int]:
    storage = tensor.untyped_storage()
    if not storage.nbytes():
        return 0, 0
    if tensor.device.type != "cpu":
        raise error(f"{whose} must be CPU resident")
    return storage.data_ptr(), storage.nbytes()


def register_spill_tensor(
    objects: object, alias_id: str, tensor: torch.Tensor, *, retain_spill_copy: bool
) -> None:
    address, size = _host_bytes(tensor, "initial object payload", PlanningError)
    objects.register_spill_bytes(  # type: ignore[attr-defined]
        alias_id, address=address, size=size, retain_spill_copy=retain_spill_copy
    )


def write_spill_tensor(objects: object, alias_id: str, tensor: torch.Tensor) -> None:
    address, size = _host_bytes(tensor, "runtime input payload", RuntimeExecutionError)
    item = _sharded_storage(objects, alias_id)
    if item is not None:
        from shadowspill.pytorch.state.storage import _write_storage_bytes

        _write_storage_bytes(
            cast("PlanObjects", objects).runtime, item, _bytes_view(tensor)
        )
        return
    objects.write_spill_bytes(alias_id, address=address, size=size)  # type: ignore[attr-defined]


def read_spill_tensor(objects: object, alias_id: str, tensor: torch.Tensor) -> None:
    address, size = _host_bytes(tensor, "writeback destination", RuntimeExecutionError)
    item = _sharded_storage(objects, alias_id)
    if item is not None:
        from shadowspill.pytorch.state.storage import _storage_bytes

        _storage_bytes(
            item,
            runtime=cast("PlanObjects", objects).runtime,
            owner=_bytes_view(tensor),
        )
        return
    objects.read_spill_bytes(alias_id, address=address, size=size)  # type: ignore[attr-defined]


def _bytes_view(tensor: torch.Tensor) -> torch.Tensor:
    return torch.empty(0, dtype=torch.uint8).set_(
        tensor.untyped_storage(), 0, (tensor.untyped_storage().nbytes(),), (1,)
    )


def _sharded_storage(objects: object, alias_id: str) -> PersistentStorage | None:
    from shadowspill.pytorch.state.registry import registry_for

    runtime = getattr(objects, "runtime", None)
    if runtime is None:
        return None
    if not cast("PlanObjects", objects).requires_storage(alias_id):
        return None
    item = registry_for(runtime).runtime_object(
        cast("PlanObjects", objects).runtime_object_id(alias_id)
    )
    return item if item is not None and item.sharding is not None else None


def spill_view(objects: object, alias_id: str) -> torch.Tensor | None:
    """One alias's bytes where they are in the spill pool, as a CPU tensor.

    A view rather than a copy: it reads the pool itself, so it is valid only
    while the runtime is idle and the object stays where it is. ``None`` where
    the bytes cannot be viewed in place; :func:`read_spill_tensor` copies them
    out then.
    """

    if _sharded_storage(objects, alias_id) is not None:
        return None
    location = objects.spill_location(alias_id)  # type: ignore[attr-defined]
    if location is None:
        return None
    address, size = location
    return cast(
        torch.Tensor,
        torch.ops.shadowspill._make_runtime_cpu_storage(
            torch.empty(0, dtype=torch.uint8, device="cpu"),
            objects.spill_pool_id,  # type: ignore[attr-defined]
            address,
            objects.runtime_object_id(alias_id),  # type: ignore[attr-defined]
            size,
        ),
    )


__all__ = [
    "read_spill_tensor",
    "register_spill_tensor",
    "spill_view",
    "write_spill_tensor",
]
