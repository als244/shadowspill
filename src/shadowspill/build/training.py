"""Build an objective and persistent gradients without optimizer computation."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch import nn

from shadowspill.artifacts.program import Program
from shadowspill.pytorch.accelerator import resolve_device
from shadowspill.pytorch.capture.retention import RetentionPolicy
from shadowspill.pytorch.partition import PartitionSpec

from .common import build_setup, compile_artifacts
from .lowering.training import lower_training
from .training_capture import capture_training_model


def build_forward_backward(
    model: nn.Module,
    *,
    objective: Callable[..., Any],
    example_args: Sequence[Any] = (),
    example_kwargs: Mapping[str, Any] | None = None,
    num_examples: int = 1,
    partition: PartitionSpec = "auto",
    operator_recompute: RetentionPolicy | None = None,
    grad_dtype: torch.dtype | None = None,
    target: str | torch.device = "cuda",
) -> Program:
    """Finalize both backward forms; order is selected later by CPU planning.

    The objective is called as objective(model, *example_args, **example_kwargs).
    Gradients persist after execution; no optimizer is captured or initialized.
    """
    device = resolve_device(target)
    assert device.index is not None
    captured = capture_training_model(
        model,
        objective,
        example_args,
        dict(example_kwargs or {}),
        num_examples=num_examples,
        partition=partition,
        operator_recompute=operator_recompute,
        gradient_dtype=grad_dtype,
        device_index=device.index,
    )
    artifacts = (
        artifact
        for example in captured.examples
        for stage in example.stages
        for variant in stage.graph_pairs.variants
        for artifact in (variant.pair.forward, variant.pair.backward)
    )
    compiled = compile_artifacts(artifacts, device)
    return lower_training(
        captured, compiled, setup=build_setup(device), gradient_dtype=grad_dtype
    )
