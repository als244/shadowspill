"""Model export and stage capture without runtime storage or an ArtifactStore."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._pytree import TreeSpec, tree_flatten

from shadowspill.pytorch.capture.artifacts import (
    GraphArtifact,
    capture_forward_stage_artifacts,
)
from shadowspill.pytorch.capture.export import ExportCapture, capture_forward
from shadowspill.pytorch.capture.fake import fake_device_inputs, fake_device_model
from shadowspill.pytorch.capture.forward import restore_module_paths, select_forward
from shadowspill.pytorch.capture.inputs import flat_runtime_arguments
from shadowspill.pytorch.guards import InputSignature, capture_input_signature
from shadowspill.pytorch.partition import (
    PartitionedExport,
    PartitionSpec,
    partition_export,
)

from .constants import literal_values
from .outputs import own_public_outputs


@dataclass(frozen=True)
class ForwardCapture:
    model: nn.Module
    exported: ExportCapture
    partitioned: PartitionedExport
    artifacts: tuple[GraphArtifact, ...]
    output_tree: TreeSpec
    input_signature: InputSignature
    output_literals: tuple[tuple[int, Any], ...]
    constant_values: Mapping[str, torch.Tensor]


def capture_forward_model(
    model: nn.Module,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    *,
    forward_fn: Callable[..., Any] | None,
    partition: PartitionSpec,
    device_index: int,
) -> ForwardCapture:
    signature = capture_input_signature((tuple(args), dict(kwargs)))
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    replica = fake_device_model(model, mode, device_index=device_index)
    view = select_forward(replica, forward_fn) if forward_fn is not None else None
    fake_args, fake_kwargs = fake_device_inputs(
        (tuple(args), dict(kwargs)), mode, device_index=device_index
    )
    with mode, torch.no_grad():
        output_values, output_tree = tree_flatten(replica(*fake_args, **fake_kwargs))
        exported = capture_forward(replica, fake_args, kwargs=fake_kwargs)
        if view is not None:
            restore_module_paths(exported, replica, view)
        constants = literal_values(exported)
        own_public_outputs(exported)
        roots = tuple(
            flat_runtime_arguments(
                exported, model, args, kwargs=kwargs, constant_values=constants
            )
        )
        partitioned = partition_export(
            exported,
            replica,
            partition=partition,
            representative_root_inputs=roots,
            derive_controls=False,
        )
        artifacts = capture_forward_stage_artifacts(partitioned)
    return ForwardCapture(
        replica,
        exported,
        partitioned,
        artifacts,
        output_tree,
        signature,
        tuple(
            (index, value)
            for index, value in enumerate(output_values)
            if not isinstance(value, torch.Tensor)
        ),
        constants,
    )
