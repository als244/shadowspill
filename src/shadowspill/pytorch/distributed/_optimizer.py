"""Compose gradient completion, a captured update, and weight publication.

The local optimizer sees ordinary tensors; the ownership policy chooses their
shapes. The captured task receives full local gradients and compute weights,
plus the owned optimizer state and optional masters.
Collectives finish in the task. No tensor arena or hidden optimizer runtime.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import ceil
from typing import cast

import torch
from torch.fx.experimental.proxy_tensor import make_fx

from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.optimizer.artifacts import (
    OptimizerCapture,
    OptimizerTensorBinding,
    OptimizerTensorRole,
)
from shadowspill.pytorch.optimizer.bindings import optimizer_input_provenance
from shadowspill.pytorch.optimizer.tasks import partition_optimizer_graph


@dataclass(frozen=True)
class UpdateLayout:
    shape: tuple[int, ...]
    dtype: torch.dtype
    group_name: str | None
    group_size: int
    group_rank: int
    master: bool = False
    gradient_group_name: str | None = None
    gradient_group_size: int | None = None
    stride: tuple[int, ...] = ()
    local_gradient: bool = True
    needs_gradient_initialization: bool = False

    @property
    def elements(self) -> int:
        return torch.Size(self.shape).numel()

    @property
    def capacity(self) -> int:
        return ceil(self.elements / self.group_size)

    @property
    def start(self) -> int:
        return self.capacity * self.group_rank

    @property
    def length(self) -> int:
        return max(0, min(self.capacity, self.elements - self.start))


def _owned(value: torch.Tensor, layout: UpdateLayout) -> torch.Tensor:
    flat = value.reshape(-1)
    # Padding is bounded by group_size - 1 for an entire logical parameter.
    padded = torch.nn.functional.pad(
        flat, (0, layout.capacity * layout.group_size - flat.numel())
    )
    return padded.narrow(0, layout.start, layout.capacity).clone()


def _gradient(
    value: torch.Tensor, layout: UpdateLayout, *, already_reduced: bool = False
) -> torch.Tensor:
    group = layout.gradient_group_name
    if already_reduced or group is None:
        return _owned(value, layout) if layout.group_size > 1 else value
    if layout.group_size > 1 and group == layout.group_name:
        flat = value.reshape(-1)
        packed = torch.nn.functional.pad(
            flat, (0, layout.capacity * layout.group_size - flat.numel())
        )
        return cast(
            torch.Tensor,
            torch.ops._c10d_functional.wait_tensor(
                torch.ops._c10d_functional.reduce_scatter_tensor(
                    packed, "sum", layout.group_size, group
                )
            ),
        )
    reduced: torch.Tensor = torch.ops._c10d_functional.wait_tensor(
        torch.ops._c10d_functional.all_reduce(value, "sum", group)
    )
    return _owned(reduced, layout) if layout.group_size > 1 else reduced


def _gather(value: torch.Tensor, layout: UpdateLayout) -> torch.Tensor:
    compute = value.to(layout.dtype).reshape(-1).contiguous()
    if layout.group_size > 1:
        compute = torch.ops._c10d_functional.wait_tensor(
            torch.ops._c10d_functional.all_gather_into_tensor(
                compute, layout.group_size, layout.group_name
            )
        )
    return compute[: layout.elements].reshape(layout.shape)


def distribute_capture(
    captured: OptimizerCapture,
    layouts: Mapping[str, UpdateLayout],
    *,
    gradient_dtype: torch.dtype | None = None,
    parameter_stage_owners: Mapping[str, tuple[int, ...]] | None = None,
    already_reduced: bool = False,
    representative_values: Mapping[str, torch.Tensor] | None = None,
) -> OptimizerCapture:
    """Re-capture the local update around reduction and owned state.

    Complete tensor shapes support general captured updates. The separate
    flat-shard policy validates elementwise independence before selecting sliced
    layouts. State tensors and update math remain optimizer-defined. All
    communication is inside these tasks.
    """
    if not isinstance(captured.update, GraphArtifact):
        raise TypeError("distributed updates require tensor-traceable optimizer tasks")
    roles = OptimizerTensorRole
    incoming = []
    provenance_values = {
        item.source: item.representative_value
        for item in captured.update.input_provenance
        if item.source is not None and item.representative_value is not None
    }
    # Preserve the current CPU storage when model Parameter.data later becomes
    # a device placeholder. Retaining the Parameter itself follows that rebind.
    provenance_values.update(
        {name: value.detach() for name, value in (representative_values or {}).items()}
    )
    for binding in captured.bindings:
        if binding.role == roles.PARAMETER:
            layout = layouts[binding.name]
            geometry = tuple(binding.tensor.shape) if layout.master else layout.shape
            dtype = binding.tensor.dtype if layout.master else layout.dtype
        elif binding.role == roles.GRADIENT:
            layout = layouts[binding.name.removeprefix("gradient.")]
            geometry, dtype = layout.shape, gradient_dtype or layout.dtype
        else:
            geometry, dtype = tuple(binding.tensor.shape), binding.tensor.dtype
        # Keep CPU control scalars on CPU and geometry tensors on their fake
        # execution device. Real parameter values are supplied by materialization.
        tensor = binding.tensor.new_empty(geometry, dtype=dtype)
        if binding.role == roles.PARAMETER and not layout.master and layout.stride:
            tensor = binding.tensor.new_empty_strided(
                layout.shape, layout.stride, dtype=dtype
            )
        incoming.append(replace(binding, tensor=tensor))
    for name, layout in layouts.items():
        if layout.master:
            incoming.append(
                OptimizerTensorBinding(
                    f"compute.{name}",
                    roles.COMPUTE_COPY,
                    next(
                        b.tensor for b in captured.bindings if b.name == name
                    ).new_empty_strided(
                        layout.shape,
                        layout.stride or _contiguous_strides(layout.shape),
                        dtype=layout.dtype,
                    ),
                    True,
                    True,
                )
            )
    positions = {b.name: i for i, b in enumerate(incoming)}
    original = captured.update.graph_module

    def update(*values: torch.Tensor) -> tuple[torch.Tensor, ...]:
        arguments: list[torch.Tensor] = []
        owned: dict[str, torch.Tensor] = {}
        for binding in captured.bindings:
            value = values[positions[binding.name]]
            if binding.role == roles.PARAMETER:
                layout = layouts[binding.name]
                value = (
                    value
                    if layout.master or layout.group_size == 1
                    else _owned(value, layout)
                )
                owned[binding.name] = value
            elif binding.role == roles.GRADIENT:
                name = binding.name.removeprefix("gradient.")
                value = (
                    _gradient(value, layouts[name], already_reduced=already_reduced)
                    .to(binding.tensor.dtype)
                    .reshape(binding.tensor.shape)
                )
            arguments.append(value)
        original(*arguments)
        for name, layout in layouts.items():
            destination = f"compute.{name}" if layout.master else name
            values[positions[destination]].copy_(_gather(owned[name], layout))
        return tuple(v for v, b in zip(values, incoming, strict=True) if b.mutable)

    examples = tuple(b.tensor for b in incoming)
    with torch.no_grad():
        graph = make_fx(update, tracing_mode="fake")(*examples)
        artifact = GraphArtifact.capture(
            kind="optimizer",
            graph_module=graph,
            example_inputs=examples,
            input_provenance=optimizer_input_provenance(
                tuple(incoming), provenance_values
            ),
        )
    tasks = partition_optimizer_graph(
        artifact, tuple(incoming), parameter_stage_owners=parameter_stage_owners
    )
    return replace(
        captured,
        update=artifact,
        bindings=tuple(incoming),
        update_tasks=tasks,
        mutation_names=tuple(b.name for b in incoming if b.mutable),
    )


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = []
    extent = 1
    for size in reversed(shape):
        strides.append(extent)
        extent *= size
    return tuple(reversed(strides))
