"""Immutable optimizer bindings and captured task artifacts."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import torch

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
    #: The weights the model computes with, where the optimizer's parameter is
    #: a master copy of them at another precision: the update writes them from
    #: their master once it has stepped.
    COMPUTE_COPY = "compute_copy"


@dataclass(frozen=True, slots=True)
class OptimizerTensorBinding:
    name: str
    role: OptimizerTensorRole
    tensor: torch.Tensor
    mutable: bool
    spillable: bool
    component_path: tuple[str, ...] = ()
    # Ownership is explicit; consumers must not parse dotted parameter names.
    parameter_name: str | None = None

    @property
    def logical_name(self) -> str:
        suffix = ".".join(self.component_path)
        return self.name[: -len(suffix) - 1] if suffix else self.name


@dataclass(frozen=True, slots=True)
class OptimizerCapture:
    """The optimizer's update, as tasks, and its explicit tensor inventory."""

    optimizer_type: str
    update: GraphArtifact
    update_tasks: tuple[OptimizerTask, ...]
    bindings: tuple[OptimizerTensorBinding, ...]
    mutation_names: tuple[str, ...]
    initialized_state_dict: dict[str, Any] | None = field(
        default=None, repr=False, compare=False
    )


@dataclass(frozen=True, slots=True)
class OptimizerTask:
    """One dependency-closed component of the optimizer update."""

    artifact: GraphArtifact
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
    "OptimizerCapture",
    "OptimizerTask",
    "OptimizerTensorBinding",
    "OptimizerTensorRole",
]
