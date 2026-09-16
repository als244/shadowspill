"""A CPU tensor's bytes, as the runtime takes them: an address and a size.

The runtime's spill objects are byte ranges. A frontend that keeps its state in
tensors has to say which bytes, and PyTorch answers with a storage pointer and a
length -- which is the whole of what `PlanObjects` ever used a tensor for. These
four are that conversion, and the residency check that goes with it: the address
handed over must be host memory, and only the framework can say whether it is.
"""

from __future__ import annotations

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


def spill_window(objects: object, alias_id: str) -> torch.Tensor | None:
    """The object's spill bytes as a tensor view, or None if they are stale."""

    window = objects.spill_window(alias_id)  # type: ignore[attr-defined]
    if window is None:
        return None
    return torch.frombuffer(window, dtype=torch.uint8)


__all__ = [
    "read_spill_tensor",
    "register_spill_tensor",
    "spill_window",
    "write_spill_tensor",
]
