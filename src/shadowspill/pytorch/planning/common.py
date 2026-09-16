"""Small shared helpers for composable PyTorch planning phases."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils._pytree import tree_flatten

from shadowspill.errors import (
    AdmissionError,
    PlanningError,
)
from shadowspill.pipeline.common import round_up

_MIB = 1 << 20
_SPILL_LEEWAY_MINIMUM = 256 * _MIB
_SPILL_ALIGNMENT = 64 << 10


def validate_cpu_model(model: nn.Module) -> None:
    """Require initialized CPU registrations before allocator materialization."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    for name, tensor in tuple(model.named_parameters()) + tuple(model.named_buffers()):
        if tensor.device.type != "cpu":
            raise PlanningError(
                f"registered tensor {name!r} must be CPU resident before planning"
            )


def estimate_spill_reservation(
    model: nn.Module,
    example_inputs: object,
    spill_budget: int,
) -> int:
    """Conservatively validate that spill storage can hold state and inputs."""

    tensors = [
        tensor
        for _name, tensor in (
            *tuple(model.named_parameters(remove_duplicate=False)),
            *tuple(model.named_buffers(remove_duplicate=False)),
        )
    ]
    leaves, _ = tree_flatten(example_inputs)
    tensors.extend(value for value in leaves if isinstance(value, torch.Tensor))
    unique: dict[tuple[str, int], int] = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        unique[(tensor.device.type, storage._cdata)] = int(storage.nbytes())
    base = sum(unique.values())
    requested = round_up(
        base + max(_SPILL_LEEWAY_MINIMUM, base // 10),
        _SPILL_ALIGNMENT,
    )
    if requested > spill_budget:
        raise AdmissionError(
            "spill-pool budget cannot hold model/input storage plus admission "
            f"leeway: required={requested}, budget={spill_budget}"
        )
    return requested


__all__ = ["estimate_spill_reservation", "validate_cpu_model"]
