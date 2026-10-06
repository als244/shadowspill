"""Register model state and root inputs for forward lowering."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

from shadowspill.ir import ObjectRole, Persistence, SharedResidencyPolicy
from shadowspill.pytorch.representations import RootInputKey, tensor_components
from shadowspill.task.slots import ObjectSlot

from ...partition import PartitionedExport
from ..catalog import (
    ObjectCatalog,
    register_model_state,
    tensor_value_role,
)
from .artifacts import ForwardObjects


def register_forward_objects(
    model: nn.Module,
    partitioned: PartitionedExport,
    *,
    device_id: str,
    shared_residency_by_root: Mapping[int, tuple[SharedResidencyPolicy, bool]]
    | None = None,
) -> ForwardObjects:
    catalog = ObjectCatalog(device_id=device_id)
    registrations, _parameter_objects = register_model_state(model, catalog)
    shared = dict(shared_residency_by_root or {})
    root_slots: list[ObjectSlot] = []
    roots: dict[RootInputKey, str] = {}
    for position, value in enumerate(partitioned.root_inputs):
        if not isinstance(value, torch.Tensor):
            continue
        for path, component in tensor_components(value):
            object_id = catalog.add(
                component,
                role=tensor_value_role(component, continuous_role=ObjectRole.INPUT),
                persistence=Persistence.STEP,
                retain_spill_copy=True,
            )
            key: RootInputKey = (position, path) if path else position
            roots[key] = object_id
            policy = shared.get(position)
            if policy is not None:
                catalog.mark_shared_residency(
                    object_id, policy[0], retain_spill_copy=policy[1]
                )
            if not path:
                root_slots.append(ObjectSlot(position, object_id))
    return ForwardObjects(
        catalog,
        registrations,
        tuple(root_slots),
        roots,
    )


__all__ = ["register_forward_objects"]
