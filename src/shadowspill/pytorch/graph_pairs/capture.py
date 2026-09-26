"""Stage-local AOT graph-pair construction."""

from __future__ import annotations

import torch
from torch.utils._pytree import tree_flatten

from shadowspill.errors import CaptureError

from ..partition.artifacts import PartitionedExport
from ..partition.differentiability import differentiable_output_positions
from .artifacts import DifferentiatedStage
from .store import GraphPairStore


def capture_training_stages(
    partitioned: PartitionedExport,
    *,
    graph_pair_store: GraphPairStore | None = None,
    accumulating: bool = False,
    gradient_dtype: torch.dtype | None = None,
    round_accumulation_once: bool = False,
) -> tuple[DifferentiatedStage, ...]:
    """Bind every stage occurrence to its structural graph pairs, their
    parameter gradients produced at ``gradient_dtype`` when one is given."""

    store = graph_pair_store or GraphPairStore()
    return tuple(
        _capture_training_stage(
            partitioned,
            index,
            graph_pair_store=store,
            accumulating=accumulating,
            gradient_dtype=gradient_dtype,
            round_accumulation_once=round_accumulation_once,
        )
        for index in range(len(partitioned.stages))
    )


def _capture_training_stage(
    partitioned: PartitionedExport,
    stage_index: int,
    *,
    graph_pair_store: GraphPairStore,
    accumulating: bool = False,
    gradient_dtype: torch.dtype | None = None,
    round_accumulation_once: bool = False,
) -> DifferentiatedStage:
    example = partitioned.stages[stage_index]
    leaves, _ = tree_flatten(example.output)
    if not leaves or any(not isinstance(value, torch.Tensor) for value in leaves):
        raise CaptureError("training stage outputs must be tensors")
    differentiable = differentiable_output_positions(example.output)
    if not differentiable:
        # The automatic partition folds a stage with nothing to differentiate
        # into the stage that consumes it, so reaching this means a supplied
        # policy drew the boundary. Say what the stage produced, because
        # "no gradient output" on its own reads as a model that cannot train.
        kinds = ", ".join(str(value.dtype).removeprefix("torch.") for value in leaves)
        raise CaptureError(
            f"training {example.stage.stage_id} has no gradient output: its "
            f"{len(leaves)} output(s) are {kinds}, which carry no gradient. A "
            "training stage is differentiated through its outputs, so every "
            "stage must produce at least one continuous value that requires "
            "one."
        )
    roots = (
        (partitioned.user_output_indices[0],)
        if stage_index == len(partitioned.stages) - 1
        else differentiable
    )
    if any(position not in differentiable for position in roots):
        raise CaptureError("terminal objective loss is not differentiable")
    return DifferentiatedStage(
        example=example,
        graph_pairs=graph_pair_store.resolve(
            example,
            roots,
            specialize_unit_tangents=stage_index == len(partitioned.stages) - 1,
            accumulating=accumulating,
            gradient_dtype=gradient_dtype,
            round_accumulation_once=round_accumulation_once,
        ),
    )


__all__ = ["capture_training_stages"]
