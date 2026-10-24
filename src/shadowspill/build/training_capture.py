"""One objective export, rebound examples, and both backward forms per stage."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode

from shadowspill.pytorch.capture.aot import (
    capture_training_objective,
    rebind_training_objective,
)
from shadowspill.pytorch.capture.fake import fake_device_inputs, fake_device_model
from shadowspill.pytorch.capture.inputs import flat_runtime_arguments
from shadowspill.pytorch.capture.retention import RetentionPolicy
from shadowspill.pytorch.graph_pairs import (
    GraphPairStore,
    PartitionedTrainingCapture,
    partition_training_capture,
)
from shadowspill.pytorch.guards import InputSignature, capture_input_signature
from shadowspill.pytorch.partition import PartitionSpec


@dataclass(frozen=True)
class TrainingCapture:
    model: nn.Module
    examples: tuple[PartitionedTrainingCapture, ...]
    input_signature: InputSignature


def capture_training_model(
    model: nn.Module,
    objective: Callable[..., Any],
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    *,
    num_examples: int,
    partition: PartitionSpec,
    operator_recompute: RetentionPolicy | None,
    gradient_dtype: torch.dtype | None,
    device_index: int,
) -> TrainingCapture:
    if isinstance(num_examples, bool) or not isinstance(num_examples, int):
        raise TypeError("num_examples must be an integer")
    if num_examples < 1:
        raise ValueError("num_examples must be positive")
    signature = capture_input_signature((tuple(args), dict(kwargs)))
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    replica = fake_device_model(model, mode, device_index=device_index)
    original = nn.Module()
    original.add_module("model", model)
    pairs = GraphPairStore()  # In-memory structural reuse; no artifact store.
    captures: list[PartitionedTrainingCapture] = []
    template = None
    with mode:
        for _ in range(num_examples):
            fake_args, fake_kwargs = fake_device_inputs(
                (tuple(args), dict(kwargs)), mode, device_index=device_index
            )
            if template is None:
                template = capture_training_objective(
                    replica, objective, fake_args, kwargs=fake_kwargs
                )
                roots = tuple(
                    flat_runtime_arguments(
                        template.exported, original, args, kwargs=kwargs
                    )
                )
            captured = rebind_training_objective(
                template, fake_args, kwargs=fake_kwargs
            )
            previous = captures[0].partitioned if captures else None
            captures.append(
                partition_training_capture(
                    captured,
                    partition=partition,
                    graph_pair_store=pairs,
                    representative_root_inputs=roots,
                    representative_stage_outputs=(
                        previous.control_values if previous is not None else None
                    ),
                    retention=operator_recompute,
                    # Even one example supports accumulating into existing gradients.
                    accumulating=True,
                    gradient_dtype=gradient_dtype,
                    template=previous,
                )
            )
    return TrainingCapture(replica, tuple(captures), signature)
