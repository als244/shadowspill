"""Bind exported inputs to original model values without allocating storage."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensor
from torch.export.graph_signature import InputKind

from shadowspill.errors import PlanningError

from .aot import ExportCapture


def flat_runtime_arguments(
    capture: ExportCapture,
    model: nn.Module,
    inputs: Sequence[Any],
    *,
    kwargs: Mapping[str, Any] | None = None,
    constant_values: Mapping[str, torch.Tensor] | None = None,
) -> list[object]:
    """Combine original registered state with one guarded user invocation."""

    flatten = getattr(capture.exported_program, "_graph_module_flat_inputs", None)
    if not callable(flatten):
        raise PlanningError("PyTorch Export flat-input adapter is unavailable")
    flat = list(flatten(tuple(inputs), dict(kwargs or {})))
    specs = capture.exported_program.graph_signature.input_specs
    if len(flat) != len(specs):
        raise PlanningError("Export flat input count changed after capture")
    for index, spec in enumerate(specs):
        if spec.kind is InputKind.PARAMETER:
            if spec.target is None:
                raise PlanningError("Export parameter input has no target")
            flat[index] = model.get_parameter(spec.target)
        elif spec.kind is InputKind.BUFFER:
            if spec.target is None:
                raise PlanningError("Export buffer input has no target")
            flat[index] = model.get_buffer(spec.target)
        elif spec.kind is InputKind.CONSTANT_TENSOR:
            if spec.target is None:
                raise PlanningError("Export constant tensor has no target")
            # Tensor literals created inside forward live in Export's constant
            # table, not on the original module. Existing attribute constants
            # may instead be fake after device capture; recover those from the
            # source module as before.
            value = (constant_values or {}).get(spec.target, flat[index])
            if isinstance(value, FakeTensor) or not isinstance(value, torch.Tensor):
                value = _resolve_attribute(model, spec.target)
            if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
                raise PlanningError(
                    f"constant tensor {spec.target!r} is not CPU resident"
                )
            flat[index] = value
    return flat


def _resolve_attribute(module: nn.Module, target: str) -> object:
    value: object = module
    for component in target.split("."):
        if not hasattr(value, component):
            raise PlanningError(f"model has no exported attribute {target!r}")
        value = getattr(value, component)
    return value
