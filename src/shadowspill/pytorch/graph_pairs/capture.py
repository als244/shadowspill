"""Stage-local AOT graph-pair construction."""

from __future__ import annotations

import torch
from torch.utils._pytree import tree_flatten

from shadowspill.errors import CaptureError

from ..partition.artifacts import PartitionedExport
from .artifacts import DifferentiatedStage
from .store import GraphPairStore


def capture_training_stages(
    partitioned: PartitionedExport,
    *,
    graph_pair_repository: GraphPairStore | None = None,
) -> tuple[DifferentiatedStage, ...]:
    """Bind every stage occurrence to its structural graph pairs."""

    cache = graph_pair_repository or GraphPairStore()
    return tuple(
        _capture_training_stage(
            partitioned,
            index,
            graph_pair_repository=cache,
        )
        for index in range(len(partitioned.stages))
    )


def _capture_training_stage(
    partitioned: PartitionedExport,
    stage_index: int,
    *,
    graph_pair_repository: GraphPairStore,
) -> DifferentiatedStage:
    example = partitioned.stages[stage_index]
    leaves, _ = tree_flatten(example.output)
    if not leaves or any(not isinstance(value, torch.Tensor) for value in leaves):
        raise CaptureError("training stage outputs must be tensors")
    differentiable = tuple(
        position
        for position, value in enumerate(leaves)
        if value.requires_grad and (value.is_floating_point() or value.is_complex())
    )
    if not differentiable:
        raise CaptureError(f"training {example.stage.stage_id} has no gradient output")
    roots = (
        (partitioned.user_output_indices[0],)
        if stage_index == len(partitioned.stages) - 1
        else differentiable
    )
    if any(position not in differentiable for position in roots):
        raise CaptureError("terminal objective loss is not differentiable")
    return DifferentiatedStage(
        example=example,
        graph_pairs=graph_pair_repository.resolve(
            example,
            roots,
            specialize_unit_tangents=stage_index == len(partitioned.stages) - 1,
        ),
    )


__all__ = ["capture_training_stages"]
