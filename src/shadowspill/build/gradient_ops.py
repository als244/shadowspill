"""Local padding and cross-invocation accumulation around distributed reductions."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch._guards import detect_fake_mode
from torch.fx.experimental.proxy_tensor import make_fx

from shadowspill.parallelism.ownership import ParameterOwnership
from shadowspill.parallelism.reductions import GradientReduction, gradient_reduction
from shadowspill.pytorch.capture.artifacts import GraphArtifact, TaskInputProvenance
from shadowspill.pytorch.optimizer.staging import training_parameters_with_gradients
from shadowspill.task.inputs import TaskInputRole

from .training_capture import TrainingCapture


@dataclass(frozen=True)
class GradientOperations:
    reduction: GradientReduction
    owned: torch.Tensor
    padding: GraphArtifact | None
    accumulation: GraphArtifact | None


def capture_gradient_operations(
    model: nn.Module,
    training: TrainingCapture,
    ownership: tuple[ParameterOwnership, ...],
    gradient_dtype: torch.dtype | None,
    *,
    destination: Literal["gradients", "optimizer"] = "gradients",
) -> dict[str, GradientOperations]:
    """Capture only local work, using the model's existing fake device mode."""
    trained = training_parameters_with_gradients(
        training.examples, dict(model.named_parameters())
    )
    mode = detect_fake_mode(training.examples[0].training.exported.flat_inputs)
    assert mode is not None
    result = {}
    with mode, torch.no_grad():
        for parameter in ownership:
            if parameter.name not in trained:
                continue
            value = training.model.get_parameter(parameter.name)
            if (
                not value.is_contiguous()
                or value.storage_offset() != 0
                or value.untyped_storage().nbytes()
                != value.numel() * value.element_size()
            ):
                raise NotImplementedError(
                    "distributed gradient lowering currently requires "
                    "contiguous whole-storage parameters"
                )
            reduction = gradient_reduction(parameter, destination=destination)
            if (
                not reduction.phases
                or reduction.phases[0].operation != "reduce_scatter"
            ):
                raise NotImplementedError(
                    "initial distributed backward lowering requires sharded gradients"
                )
            dtype = gradient_dtype or value.dtype
            gradient = torch.empty_like(value, dtype=dtype)
            padding = _padding(gradient, reduction.padded_elements)
            owned = torch.empty(
                (reduction.result_elements,), dtype=dtype, device=value.device
            )
            addition = _accumulation(owned) if destination == "gradients" else None
            result[parameter.name] = GradientOperations(
                reduction, owned, padding, addition
            )
    return result


def _padding(gradient: torch.Tensor, capacity: int) -> GraphArtifact | None:
    extra = capacity - gradient.numel()
    if extra == 0:
        return None

    def pad(value: torch.Tensor) -> tuple[torch.Tensor]:
        return (torch.nn.functional.pad(value.reshape(-1), (0, extra)),)

    return _capture(pad, (gradient,))


def _accumulation(owned: torch.Tensor) -> GraphArtifact:
    def add(existing: torch.Tensor, new: torch.Tensor) -> tuple[()]:
        existing.add_(new)
        return ()

    return _capture(add, (owned, torch.empty_like(owned)))


def _capture(
    function: Callable[..., object], inputs: tuple[torch.Tensor, ...]
) -> GraphArtifact:
    graph = make_fx(function, tracing_mode="fake")(*inputs)
    return GraphArtifact.capture(
        kind="backward",
        graph_module=graph,
        example_inputs=inputs,
        input_provenance=tuple(
            TaskInputProvenance(TaskInputRole.GRADIENT) for _ in inputs
        ),
    )
