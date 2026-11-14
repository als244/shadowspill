"""Training objective export and rebinding without whole-model differentiation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.nn as nn
from torch.export.graph_signature import InputKind, OutputKind
from torch.utils._pytree import tree_flatten

from shadowspill.errors import CaptureError, ObjectiveError
from shadowspill.pytorch.capture.artifacts import (
    AotGraphPair,
    ObjectiveSchema,
    capture_objective_schema,
    normalize_objective_result,
)
from shadowspill.pytorch.contracts import ObjectiveResult

from .aot import _capture_pair
from .export import ExportCapture, _export, _export_mutations, _flatten_inputs


@dataclass(frozen=True, slots=True)
class TrainingObjectiveCapture:
    """Exported objective semantics before stage-local differentiation."""

    exported: ExportCapture
    capture_module: nn.Module
    objective_schema: ObjectiveSchema


@dataclass(frozen=True, slots=True)
class TrainingCapture(TrainingObjectiveCapture):
    """Objective export plus whole-graph AOT alternatives for oracle use."""

    save_pair: AotGraphPair
    recompute_pair: AotGraphPair


class _ObjectiveModule(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        objective: Callable[..., torch.Tensor | ObjectiveResult],
        schema: ObjectiveSchema,
    ) -> None:
        super().__init__()
        self.model = model
        self.objective = objective
        self.schema = schema

    def forward(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, ...]:
        loss, metrics = normalize_objective_result(
            self.objective(self.model, *args, **kwargs), require_grad=False
        )
        leaves, tree_spec = tree_flatten(metrics)
        if tree_spec != self.schema.metric_tree_spec:
            raise ObjectiveError("objective metric structure changed during capture")
        tensor_metrics = tuple(
            leaves[position].detach()
            for position in self.schema.tensor_metric_positions
        )
        return (loss, *tensor_metrics)


def capture_training(
    model: nn.Module,
    objective: Callable[..., torch.Tensor | ObjectiveResult],
    microbatch: Sequence[Any],
) -> TrainingCapture:
    """Capture the objective plus its whole-graph ``save`` and ``recompute`` pairs."""

    objective_capture = capture_training_objective(model, objective, microbatch)
    save_pair = _capture_pair(objective_capture.exported, memory_budget=1.0)
    recompute_pair = _capture_pair(objective_capture.exported, memory_budget=0.0)
    return TrainingCapture(
        exported=objective_capture.exported,
        capture_module=objective_capture.capture_module,
        objective_schema=objective_capture.objective_schema,
        save_pair=save_pair,
        recompute_pair=recompute_pair,
    )


def capture_training_objective(
    model: nn.Module,
    objective: Callable[..., torch.Tensor | ObjectiveResult],
    microbatch: Sequence[Any],
    *,
    kwargs: Mapping[str, Any] | None = None,
) -> TrainingObjectiveCapture:
    """Export an objective without constructing an unused whole-model VJP."""

    try:
        probe_loss, probe_metrics = normalize_objective_result(
            objective(model, *microbatch, **dict(kwargs or {})), require_grad=True
        )
        del probe_loss
        schema = capture_objective_schema(probe_metrics)
        capture_module = _ObjectiveModule(model, objective, schema)
        exported = _export(capture_module, microbatch, kwargs)
    except (CaptureError, ObjectiveError):
        raise
    except BaseException as error:
        raise CaptureError(f"training objective capture failed: {error}") from error
    return TrainingObjectiveCapture(
        exported=exported,
        capture_module=capture_module,
        objective_schema=schema,
    )


def rebind_training_objective(
    capture: TrainingObjectiveCapture,
    microbatch: Sequence[Any],
    *,
    kwargs: Mapping[str, Any] | None = None,
) -> TrainingObjectiveCapture:
    """The same exported objective bound to another position's example inputs.

    Positions with one input structure share the export; only the flattened
    example arguments differ, and those are cheap.
    """

    exported = replace(
        capture.exported,
        flat_inputs=_flatten_inputs(
            capture.exported.exported_program, microbatch, kwargs
        ),
    )
    return replace(capture, exported=exported)


def bind_training_objective(
    program: torch.export.ExportedProgram,
    model: nn.Module,
    schema: ObjectiveSchema,
    microbatch: Sequence[Any],
) -> TrainingObjectiveCapture:
    """Bind an explicitly loaded export to fresh fake state without exporting."""

    for spec in program.graph_signature.input_specs:
        if spec.kind not in {InputKind.PARAMETER, InputKind.BUFFER}:
            continue
        if not isinstance(spec.target, str) or not spec.target.startswith("model."):
            raise CaptureError("prepared training state target is not rooted at model")
        name = spec.target.removeprefix("model.")
        # Export stores non-persistent buffers alongside lifted constants, while
        # persistent buffers and parameters live in state_dict. Both kinds of
        # buffer must bind to this model, never to the archived meta placeholder.
        state = (
            program.constants
            if spec.kind is InputKind.BUFFER and not spec.persistent
            else program.state_dict
        )
        state[spec.target] = (
            model.get_parameter(name)
            if spec.kind is InputKind.PARAMETER
            else model.get_buffer(name)
        )
    # Partition policies inspect the source module hierarchy. The loaded graph
    # is the objective; no Python objective is called during admission.
    module = nn.Module()
    module.add_module("model", model)
    return TrainingObjectiveCapture(
        ExportCapture(
            program,
            _flatten_inputs(program, microbatch),
            tuple(
                i
                for i, spec in enumerate(program.graph_signature.output_specs)
                if spec.kind is OutputKind.USER_OUTPUT
            ),
            _export_mutations(program.graph_signature),
        ),
        module,
        schema,
    )
