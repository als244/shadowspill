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

from shadowspill.pytorch.capture.artifacts import GraphArtifact


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
    optimizer: torch.optim.Optimizer = field(repr=False, compare=False)

    @classmethod
    def capture(
        cls,
        optimizer: torch.optim.Optimizer,
        bindings: tuple[OptimizerTensorBinding, ...],
    ) -> OpaqueOptimizerArtifact:
        optimizer_type = f"{type(optimizer).__module__}.{type(optimizer).__qualname__}"
        step = inspect.unwrap(type(optimizer).step)
        code = getattr(step, "__code__", None)
        code_identity = (
            None
            if code is None
            else {
                "bytecode": code.co_code.hex(),
                "constants": tuple(repr(value) for value in code.co_consts),
                "names": code.co_names,
            }
        )
        identity = {
            "kind": "opaque_optimizer",
            # The profiling executable must restore captured gradients onto
            # its copied Parameters.  Version that construction contract here
            # so correcting it invalidates only opaque-optimizer profiles,
            # rather than every compiled graph profile in the cache.
            "profiling_contract": "representative_gradients/v1",
            "optimizer_type": optimizer_type,
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
            "cuda": torch.version.cuda,
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        return cls(
            optimizer_type,
            hashlib.sha256(encoded.encode()).hexdigest(),
            optimizer,
        )


OptimizerTaskArtifact = GraphArtifact | OpaqueOptimizerArtifact


@dataclass(frozen=True, slots=True)
class OptimizerCapture:
    """First/recurrent optimizer task semantics and explicit tensor inventory."""

    optimizer_type: str
    first_step_is_opaque: bool
    created_state_names: tuple[str, ...]
    recurrent: OptimizerTaskArtifact | None
    recurrent_tasks: tuple[OptimizerTask, ...]
    bindings: tuple[OptimizerTensorBinding, ...]
    mutation_names: tuple[str, ...]
    preinitialized_state_names: tuple[str, ...] = ()
    opaque_reason: str | None = None
    initialized_state_dict: dict[str, Any] | None = field(
        default=None, repr=False, compare=False
    )

    @property
    def recurrent_is_opaque(self) -> bool:
        return self.recurrent is None or isinstance(
            self.recurrent, OpaqueOptimizerArtifact
        )


@dataclass(frozen=True, slots=True)
class OptimizerTask:
    """One dependency-closed recurrent optimizer component."""

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
