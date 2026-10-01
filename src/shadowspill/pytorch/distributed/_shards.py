"""Ordinary optimizer tensors sized by explicit parameter ownership."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Set
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch import nn

from ._optimizer import UpdateLayout

if TYPE_CHECKING:
    from . import BoundDistributed


def parameter_layouts(
    bound: BoundDistributed,
    weights: Mapping[str, nn.Parameter],
    receives_gradient: Set[str],
    *,
    master_dtype: torch.dtype | None,
) -> dict[str, UpdateLayout]:
    active = bound.control.exchange(
        "optimizer/active_parameters", sorted(receives_gradient)
    )
    by_rank = dict(zip(bound.control.members, map(set, active), strict=True))
    layouts = {}
    for record in bound.parameters:
        present = [record.name in by_rank[rank] for rank in record.replicas]
        if not any(present):
            continue
        weight = weights[record.name]
        replica_group, gradient_group = bound.parameter_groups[record.name]
        size = len(record.replicas) if bound.shard_optimizer else 1
        alias = (
            None
            if replica_group is None
            else bound.groups.source_names[replica_group.group_name]
        )
        gradient_alias = (
            None
            if gradient_group is None
            else bound.groups.source_names[gradient_group.group_name]
        )
        layouts[record.name] = UpdateLayout(
            tuple(weight.shape),
            weight.dtype,
            alias,
            size,
            record.replicas.index(dist.get_rank()) if size > 1 else 0,
            master_dtype is not None and master_dtype != weight.dtype,
            gradient_alias,
            len(record.contributions),
            tuple(weight.stride()),
            record.name in receives_gradient,
            not all(present),
        )
    return layouts


def optimizer_parameters(
    weights: Mapping[str, nn.Parameter],
    layouts: Mapping[str, UpdateLayout],
    *,
    master_dtype: torch.dtype | None,
) -> dict[str, nn.Parameter]:
    """Non-master owned parameters are geometry, not an extra persistent copy.

    The captured task derives its temporary owned weights from the compute
    parameter. Optional masters are real owned tensors imported with moments.
    """
    result = {}
    for name, weight in weights.items():
        layout = layouts.get(name)
        if layout is None:
            result[name] = weight
            continue
        if layout.group_size == 1 and not layout.master:
            result[name] = weight
            continue
        shape = (layout.capacity,) if layout.group_size > 1 else layout.shape
        result[name] = nn.Parameter(
            torch.empty(
                shape,
                dtype=master_dtype if layout.master else weight.dtype,
                device="cpu" if layout.master else "meta",
            ),
            requires_grad=weight.requires_grad,
        )
    return result


def validate_optimizer(
    optimizer: torch.optim.Optimizer, layouts: Mapping[str, UpdateLayout]
) -> None:
    if not any(layout.group_size > 1 for layout in layouts.values()):
        return
    known = type(optimizer) in {torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD}
    declared = getattr(optimizer, "supports_flat_parameter_shards", False) is True
    if not known and not declared:
        raise TypeError(
            "the current flat-shard ownership policy requires independent "
            "elementwise updates; this optimizer has not declared that optional "
            "capability. Use shard_optimizer=False to retain complete tensor "
            "shapes. This restriction belongs to the ownership policy, not "
            "the general optimizer interface"
        )


def fill_parameter(
    destination: torch.Tensor,
    source: torch.Tensor,
    layout: UpdateLayout,
    *,
    chunk_bytes: int = 16 << 20,
) -> None:
    """Copy an owned logical slice without flattening a noncontiguous full weight."""
    target = destination.view(-1)
    target.zero_()
    start = layout.start if layout.group_size > 1 else 0
    length = layout.length if layout.group_size > 1 else source.numel()
    elements = max(
        1, chunk_bytes // max(source.element_size(), destination.element_size())
    )
    for offset in range(0, length, elements):
        count = min(elements, length - offset)
        indices = torch.arange(
            start + offset, start + offset + count, device=source.device
        )
        target.narrow(0, offset, count).copy_(torch.take(source, indices))


def initializers(
    weights: Mapping[str, nn.Parameter],
    layouts: Mapping[str, UpdateLayout],
) -> dict[str, Callable[[torch.Tensor], None]]:
    def make(name: str) -> Callable[[torch.Tensor], None]:
        def initialize(destination: torch.Tensor) -> None:
            fill_parameter(destination, weights[name], layouts[name])

        return initialize

    return {name: make(name) for name in layouts}


def common_stage_owners(
    bound: BoundDistributed,
    local: Mapping[str, tuple[int, ...]],
) -> dict[str, tuple[int, ...]]:
    records = bound.control.exchange("optimizer/stage_owners", local)
    peers = dict(zip(bound.control.members, records, strict=True))
    return {
        parameter.name: tuple(
            sorted(
                {
                    stage
                    for rank in parameter.replicas
                    for stage in peers[rank].get(parameter.name, ())
                }
            )
        )
        for parameter in bound.parameters
    }
