"""Pure observations of weights and final gradients before each update.

Each reduced observation precedes its optimizer component without retaining
parameters or gradients from other update components.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._pytree import tree_flatten

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.artifacts import (
    GraphArtifact,
    capture_objective_schema,
)
from shadowspill.pytorch.capture.fake import fake_device_inputs

from .artifacts import OptimizerCapture, OptimizerTask, OptimizerTensorRole

ParameterMetrics = Callable[[torch.Tensor, torch.Tensor], Any]


def with_parameter_metrics(
    captured: OptimizerCapture,
    observer: ParameterMetrics | None,
    *,
    device_index: int = 0,
) -> OptimizerCapture:
    """Read compute weights and accumulated gradients before optimizer casts.

    When an optimizer holds masters, observe the model's compute copy. Results
    are detached, and the observer must not mutate either argument.
    """

    if observer is None:
        return captured
    if not callable(observer):
        raise TypeError("parameter_metrics must be callable or None")
    bindings = {item.name: item for item in captured.bindings}
    observed: set[str] = set()
    tasks = []
    for update in captured.update_tasks:
        gradients = tuple(
            name
            for name in update.binding_names
            if bindings[name].role is OptimizerTensorRole.GRADIENT
            and name not in observed
        )
        if gradients:
            pairs = []
            for gradient in gradients:
                parameter = gradient.removeprefix("gradient.")
                weight = (
                    "compute." + parameter
                    if "compute." + parameter in bindings
                    else parameter
                )
                pairs.append((weight, gradient))
            names = tuple(name for pair in pairs for name in pair)
            arguments = tuple(bindings[name].tensor for name in names)
            tasks.append(
                _capture_observer(
                    observer,
                    names,
                    gradients,
                    arguments,
                    update.completion_stage_index,
                    device_index=device_index,
                )
            )
            observed.update(gradients)
        tasks.append(update)
    return replace(captured, update_tasks=tuple(tasks))


def _capture_observer(
    observer: ParameterMetrics,
    names: tuple[str, ...],
    gradients: tuple[str, ...],
    arguments: tuple[torch.Tensor, ...],
    stage: int | None,
    *,
    device_index: int,
) -> OptimizerTask:
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    arguments = fake_device_inputs(arguments, mode, device_index=device_index)

    def metrics(*values: torch.Tensor) -> dict[str, Any]:
        return {
            name.removeprefix("gradient."): observer(values[2 * i], values[2 * i + 1])
            for i, name in enumerate(gradients)
        }

    with mode, torch.no_grad():
        schema = capture_objective_schema(metrics(*arguments))

        def flattened(*values: torch.Tensor) -> tuple[torch.Tensor, ...]:
            leaves, spec = tree_flatten(metrics(*values))
            if spec != schema.metric_tree_spec:
                raise CaptureError("parameter metric structure changed during capture")
            return tuple(
                leaves[index].detach().clone()
                for index in schema.tensor_metric_positions
            )

        graph = make_fx(flattened, tracing_mode="fake", _allow_non_fake_inputs=True)(
            *arguments
        )
        artifact = GraphArtifact.capture(
            kind="optimizer", graph_module=graph, example_inputs=arguments
        )
    if artifact.storage_contract.mutations:
        raise CaptureError("parameter_metrics must not mutate its inputs")
    if not schema.tensor_metric_positions:
        raise CaptureError("parameter_metrics must return at least one tensor")
    return OptimizerTask(artifact, names, (), stage, schema)
