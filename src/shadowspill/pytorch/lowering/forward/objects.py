"""Register model state and root inputs for forward lowering."""

from __future__ import annotations

import torch
import torch.nn as nn

from shadowspill.ir import ObjectRole, Persistence

from ...partition import PartitionedExport
from ..catalog import (
    ObjectCatalog,
    TensorSlot,
    register_model_state,
    tensor_value_role,
)
from .artifacts import ForwardObjects


def register_forward_objects(
    model: nn.Module,
    partitioned: PartitionedExport,
    *,
    device_id: str,
) -> ForwardObjects:
    catalog = ObjectCatalog(device_id=device_id)
    registrations, _parameter_objects = register_model_state(model, catalog)
    root_slots = tuple(
        TensorSlot(
            position,
            catalog.add(
                value,
                role=tensor_value_role(value, continuous_role=ObjectRole.INPUT),
                persistence=Persistence.STEP,
                retain_spill_copy=True,
            ),
        )
        for position, value in enumerate(partitioned.root_inputs)
        if isinstance(value, torch.Tensor)
    )
    return ForwardObjects(
        catalog,
        registrations,
        root_slots,
        {slot.leaf_index: slot.object_id for slot in root_slots},
    )


__all__ = ["register_forward_objects"]
