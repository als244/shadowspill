"""Register model parameter and buffer identities."""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from shadowspill.ir import (
    ObjectRole,
    Persistence,
)
from shadowspill.pytorch.representations import tensor_components

from .objects import ObjectCatalog


@dataclass(frozen=True, slots=True)
class RegistrationBinding:
    """Original registered name associated with one logical tensor view."""

    name: str
    object_id: str
    parameter: bool
    component_path: tuple[str, ...] = ()


def register_model_state(
    model: nn.Module,
    catalog: ObjectCatalog,
) -> tuple[tuple[RegistrationBinding, ...], dict[str, str]]:
    """Register parameters and buffers once for every lowering mode."""

    registrations: list[RegistrationBinding] = []
    parameter_objects: dict[str, str] = {}
    checkpoint_names = set(model.state_dict(keep_vars=True))
    for name, parameter in model.named_parameters(remove_duplicate=False):
        for path, component in tensor_components(parameter):
            object_id = catalog.add(
                component,
                role=ObjectRole.PARAMETER,
                persistence=Persistence.CHECKPOINT,
                retain_spill_copy=True,
            )
            registrations.append(RegistrationBinding(name, object_id, True, path))
            parameter_objects.setdefault(name, object_id)
    for name, buffer in model.named_buffers(remove_duplicate=False):
        for path, component in tensor_components(buffer):
            object_id = catalog.add(
                component,
                role=ObjectRole.BUFFER,
                persistence=(
                    Persistence.CHECKPOINT
                    if name in checkpoint_names
                    else Persistence.RUN
                ),
                retain_spill_copy=True,
            )
            registrations.append(RegistrationBinding(name, object_id, False, path))
    return tuple(registrations), parameter_objects
