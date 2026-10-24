"""Local padding and cross-invocation accumulation around distributed reductions."""

from collections.abc import Callable
from dataclasses import dataclass

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
    dtype: torch.dtype
    padding: GraphArtifact | None
    accumulation: GraphArtifact


def capture_gradient_operations(
    model: nn.Module,
    training: TrainingCapture,
    ownership: tuple[ParameterOwnership, ...],
    gradient_dtype: torch.dtype | None,
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
            if not value.is_contiguous():
                raise NotImplementedError(
                    "distributed gradient lowering currently requires "
                    "contiguous parameters"
                )
            reduction = gradient_reduction(parameter)
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
            addition = _accumulation(owned)
            result[parameter.name] = GradientOperations(
                reduction, dtype, padding, addition
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
