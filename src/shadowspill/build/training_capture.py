"""Export each input signature once and build both backward forms per stage."""

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

from .constants import literal_values
from .outputs import own_public_outputs
from .shared_gradients import include_shared_gradients


@dataclass(frozen=True)
class TrainingCapture:
    model: nn.Module
    examples: tuple[PartitionedTrainingCapture, ...]
    input_signatures: tuple[InputSignature, ...]
    constant_values: tuple[Mapping[str, torch.Tensor], ...]


def capture_training_model(
    model: nn.Module,
    objective: Callable[..., Any],
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    *,
    num_examples: int,
    example_calls: Sequence[tuple[Sequence[Any], Mapping[str, Any]]] | None,
    fuse_narrow_gradient_accumulation: bool,
    partition: PartitionSpec,
    operator_recompute: RetentionPolicy | None,
    gradient_dtype: torch.dtype | None,
    device_index: int,
    gradient_shards: Mapping[str, int] | None = None,
) -> TrainingCapture:
    if isinstance(num_examples, bool) or not isinstance(num_examples, int):
        raise TypeError("num_examples must be an integer")
    if num_examples < 1:
        raise ValueError("num_examples must be positive")
    if example_calls is not None:
        if args or kwargs or num_examples != 1:
            raise ValueError(
                "example_calls cannot be combined with "
                "example_args/kwargs or num_examples"
            )
        if not example_calls:
            raise ValueError("example_calls must not be empty")
        calls = [(tuple(a), dict(k)) for a, k in example_calls]
    else:
        calls = [(tuple(args), dict(kwargs))] * num_examples
    signatures = tuple(capture_input_signature(call) for call in calls)
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    replica = fake_device_model(model, mode, device_index=device_index)
    original = nn.Module()
    original.add_module("model", model)
    pairs = GraphPairStore(gradient_shards=gradient_shards)
    captures = []
    constants_by_example = []
    templates: dict[
        str, tuple[Any, dict[str, torch.Tensor], tuple[Any, ...], Any]
    ] = {}
    with mode:
        for (args, kwargs), signature in zip(calls, signatures, strict=True):
            fake_args, fake_kwargs = fake_device_inputs(
                (args, kwargs), mode, device_index=device_index
            )
            saved = templates.get(signature.digest)
            if saved is None:
                template = capture_training_objective(
                    replica, objective, fake_args, kwargs=fake_kwargs
                )
                constants = literal_values(template.exported)
                own_public_outputs(template.exported)
                roots = tuple(
                    flat_runtime_arguments(
                        template.exported,
                        original,
                        args,
                        kwargs=kwargs,
                        constant_values=constants,
                    )
                )
                previous = None
            else:
                template, constants, roots, previous = saved
            captured = rebind_training_objective(
                template, fake_args, kwargs=fake_kwargs
            )
            partitioned = partition_training_capture(
                captured,
                partition=partition,
                graph_pair_store=pairs,
                representative_root_inputs=roots,
                representative_stage_outputs=(
                    previous.control_values if previous is not None else None
                ),
                retention=operator_recompute,
                accumulating=True,
                gradient_dtype=gradient_dtype,
                fuse_narrow_gradient_accumulation=fuse_narrow_gradient_accumulation,
                template=previous,
            )
            templates[signature.digest] = (
                template,
                constants,
                roots,
                partitioned.partitioned,
            )
            captures.append(
                include_shared_gradients(
                    partitioned,
                    fuse_narrow_gradient_accumulation=fuse_narrow_gradient_accumulation,
                )
            )
            constants_by_example.append(constants)
    return TrainingCapture(
        replica, tuple(captures), signatures, tuple(constants_by_example)
    )
