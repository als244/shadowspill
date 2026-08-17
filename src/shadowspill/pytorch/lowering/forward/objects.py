"""Register model state and root inputs for forward lowering."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

from shadowspill.ir import ObjectRole, Persistence, SharedResidencyPolicy

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
    shared_residency_by_root: Mapping[
        int, tuple[SharedResidencyPolicy, bool]
    ] | None = None,
) -> ForwardObjects:
    catalog = ObjectCatalog(device_id=device_id)
    registrations, _parameter_objects = register_model_state(model, catalog)
    shared = dict(shared_residency_by_root or {})
    root_slots: list[TensorSlot] = []
    for position, value in enumerate(partitioned.root_inputs):
        if not isinstance(value, torch.Tensor):
            continue
        object_id = catalog.add(
            value,
            role=tensor_value_role(value, continuous_role=ObjectRole.INPUT),
            persistence=Persistence.STEP,
            retain_spill_copy=True,
        )
        policy = shared.get(position)
        if policy is not None:
            catalog.mark_shared_residency(
                object_id,
                policy[0],
                retain_spill_copy=policy[1],
            )
        root_slots.append(TensorSlot(position, object_id))
    return ForwardObjects(
        catalog,
        registrations,
        tuple(root_slots),
        {slot.leaf_index: slot.object_id for slot in root_slots},
    )


__all__ = ["register_forward_objects"]
