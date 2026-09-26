"""A CPU tensor's bytes, as the runtime takes them: an address and a size.

The runtime's spill objects are byte ranges. A frontend that keeps its state in
tensors has to say which bytes, and PyTorch answers with a storage pointer and a
length -- which is the whole of what `PlanObjects` ever used a tensor for. These
three are that conversion, and the residency check that goes with it: the address
handed over must be host memory, and only the framework can say whether it is.
"""

from __future__ import annotations

from typing import cast

import torch

from shadowspill.errors import PlanningError
from shadowspill.runtime.failures import RuntimeExecutionError


def _host_bytes(
    tensor: torch.Tensor, whose: str, error: type[Exception]
) -> tuple[int, int]:
    if tensor.device.type != "cpu":
        raise error(f"{whose} must be CPU resident")
    storage = tensor.untyped_storage()
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
    objects.write_spill_bytes(alias_id, address=address, size=size)  # type: ignore[attr-defined]


def read_spill_tensor(objects: object, alias_id: str, tensor: torch.Tensor) -> None:
    address, size = _host_bytes(tensor, "writeback destination", RuntimeExecutionError)
    objects.read_spill_bytes(alias_id, address=address, size=size)  # type: ignore[attr-defined]


def spill_view(objects: object, alias_id: str) -> torch.Tensor | None:
    """One alias's bytes where they are in the spill pool, as a CPU tensor.

    A view rather than a copy: it reads the pool itself, so it is valid only
    while the runtime is idle and the object stays where it is. ``None`` where
    the bytes cannot be viewed in place; :func:`read_spill_tensor` copies them
    out then.
    """

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
