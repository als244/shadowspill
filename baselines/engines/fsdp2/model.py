"""The shared training model and FSDP2's sharded initialization."""

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from baselines.common.training import (
    TrainingModel,
    canonical_name,
    materialize,
    pinned_parameters,
    retie,
    shared_parameter_groups,
)


@torch.no_grad()
def initialize(module: nn.Module) -> None:
    """Reset each module, staging offloaded DTensors on their mesh device.

    DTensor RNG supports accelerator storage only. A CPU tensor on a CUDA
    mesh enters the CUDA RNG context, which restores the CPU RNG after every
    operation: equal-shaped weights then get identical random values.
    Initialize one module's shards at a time on the accelerator and copy
    back into the original CPU parameters, preserving FSDP's storage and
    pinning. The DTensors retain their global shapes and shard placements.
    """

    for child in module.modules():
        reset = getattr(child, "reset_parameters", None)
        if not callable(reset):
            continue
        originals = {
            name: parameter
            for name, parameter in child.named_parameters(
                recurse=False, remove_duplicate=False
            )
            if isinstance(parameter, DTensor)
            and parameter.to_local().device.type == "cpu"
        }
        staged = {
            parameter: nn.Parameter(
                parameter.to(parameter.device_mesh.device_type, copy=True),
                requires_grad=parameter.requires_grad,
            )
            for parameter in originals.values()
        }
        try:
            for name, parameter in originals.items():
                setattr(child, name, staged[parameter])
            reset()
            for parameter in staged:
                parameter.to_local().copy_(staged[parameter].to_local())
        finally:
            for name, parameter in originals.items():
                setattr(child, name, parameter)
            staged.clear()


__all__ = [
    "TrainingModel",
    "canonical_name",
    "initialize",
    "materialize",
    "pinned_parameters",
    "retie",
    "shared_parameter_groups",
]
