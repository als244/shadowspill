"""Immutable optimizer bindings and captured task artifacts."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import torch

from shadowspill.pytorch.accelerator import provider_version
from shadowspill.pytorch.capture.artifacts import GraphArtifact


def optimizer_type_name(optimizer: torch.optim.Optimizer) -> str:
    return f"{type(optimizer).__module__}.{type(optimizer).__qualname__}"


def code_identity(function: object) -> dict[str, Any] | None:
    """What a callable does, as its code: bytecode, constants, names.

    None when the callable has no Python code object to read, such as a
    builtin or a callable class instance without one.
    """

    if not callable(function):
        return None
    target = inspect.unwrap(function)
    code = getattr(target, "__code__", None)
    if code is None:
        code = getattr(inspect.unwrap(type(target).__call__), "__code__", None)
    if code is None:
        return None
    return {
        "bytecode": code.co_code.hex(),
        "constants": tuple(repr(value) for value in code.co_consts),
        "names": code.co_names,
    }


def optimizer_step_identity(
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any] | None:
    """What the optimizer's step does, as its code."""

    return code_identity(type(optimizer).step)


class OptimizerTensorRole(StrEnum):
    PARAMETER = "parameter"
    GRADIENT = "gradient"
    STATE = "state"
    HYPERPARAMETER = "hyperparameter"


@dataclass(frozen=True, slots=True)
class OptimizerTensorBinding:
    name: str
    role: OptimizerTensorRole
    tensor: torch.Tensor
    mutable: bool
    spillable: bool


@dataclass(frozen=True, slots=True)
class OpaqueOptimizerArtifact:
    """One eager optimizer task with a deterministic structural identity."""

    optimizer_type: str
    compatibility_digest: str
    parameter_names: tuple[str | None, ...]
    optimizer: torch.optim.Optimizer = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.optimizer_type:
            raise ValueError("opaque optimizer type must be non-empty")
        if len(self.compatibility_digest) != 64:
            raise ValueError("opaque optimizer digest must be SHA-256")
        named_parameters = tuple(
            name for name in self.parameter_names if name is not None
        )
        if not self.parameter_names or any(name == "" for name in named_parameters):
            raise ValueError("opaque optimizer parameter inventory is invalid")
        if len(named_parameters) != len(set(named_parameters)):
            raise ValueError("opaque optimizer parameter names must be unique")

    @classmethod
    def capture(
        cls,
        optimizer: torch.optim.Optimizer,
        bindings: tuple[OptimizerTensorBinding, ...],
    ) -> OpaqueOptimizerArtifact:
        optimizer_type = optimizer_type_name(optimizer)
        code_identity = optimizer_step_identity(optimizer)
        parameter_name_by_id = {
            id(binding.tensor): binding.name
            for binding in bindings
            if binding.role is OptimizerTensorRole.PARAMETER
        }
        parameter_names = tuple(
            parameter_name_by_id.get(id(parameter))
            for group in optimizer.param_groups
            for parameter in group["params"]
        )
        identity = {
            "kind": "opaque_optimizer",
            # The profiling executable must restore captured gradients onto
            # its copied Parameters.  Version that construction contract here
            # so correcting it invalidates only opaque-optimizer profiles,
            # rather than every compiled graph profile in the cache.
            "profiling_contract": "update_with_restored_gradients/v1",
            "optimizer_type": optimizer_type,
            "parameter_names": parameter_names,
            "step": code_identity,
            "bindings": [
                {
                    "name": binding.name,
                    "role": binding.role.value,
                    "mutable": binding.mutable,
                    "spillable": binding.spillable,
                    "shape": tuple(binding.tensor.shape),
                    "stride": tuple(binding.tensor.stride()),
                    "dtype": str(binding.tensor.dtype),
                    "device": binding.tensor.device.type,
                }
                for binding in bindings
            ],
            "groups": [
                optimizer_value_identity(
                    {key: value for key, value in group.items() if key != "params"}
                )
                for group in optimizer.param_groups
            ],
            "torch": torch.__version__,
            "provider": provider_version(),
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        return cls(
            optimizer_type,
            hashlib.sha256(encoded.encode()).hexdigest(),
            parameter_names,
            optimizer,
        )


OptimizerTaskArtifact = GraphArtifact | OpaqueOptimizerArtifact


@dataclass(frozen=True, slots=True)
class OptimizerCapture:
    """The optimizer's update, as tasks, and its explicit tensor inventory."""

    optimizer_type: str
    update: OptimizerTaskArtifact | None
    update_tasks: tuple[OptimizerTask, ...]
    bindings: tuple[OptimizerTensorBinding, ...]
    mutation_names: tuple[str, ...]
    opaque_reason: str | None = None
    initialized_state_dict: dict[str, Any] | None = field(
        default=None, repr=False, compare=False
    )

    @property
    def update_is_opaque(self) -> bool:
        return self.update is None or isinstance(self.update, OpaqueOptimizerArtifact)


@dataclass(frozen=True, slots=True)
class OptimizerTask:
    """One dependency-closed component of the optimizer update."""

    artifact: OptimizerTaskArtifact
    binding_names: tuple[str, ...]
    mutation_names: tuple[str, ...]
    completion_stage_index: int | None = None


def optimizer_value_identity(value: Any) -> Any:
    """Serialize bounded optimizer options without retaining framework values."""

    if isinstance(value, torch.Tensor):
        return {
            "tensor": {
                "shape": tuple(value.shape),
                "stride": tuple(value.stride()),
                "dtype": str(value.dtype),
                "device": value.device.type,
            }
        }
    if isinstance(value, Mapping):
        return {
            str(key): optimizer_value_identity(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, tuple):
        return {"tuple": [optimizer_value_identity(item) for item in value]}
    if isinstance(value, list):
        return {"list": [optimizer_value_identity(item) for item in value]}
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return {"type": type(value).__qualname__, "value": repr(value)}


__all__ = [
    "OpaqueOptimizerArtifact",
    "OptimizerCapture",
    "OptimizerTask",
    "OptimizerTaskArtifact",
    "OptimizerTensorBinding",
    "OptimizerTensorRole",
]
