"""Stage-local AOT graph-pair construction."""

from __future__ import annotations

import torch
from torch.utils._pytree import tree_flatten

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.retention import RetentionPolicy

from ..partition.artifacts import PartitionedExport
from ..partition.differentiability import differentiable_output_positions
from .artifacts import DifferentiatedStage
from .store import GraphPairStore


def capture_training_stages(
    partitioned: PartitionedExport,
    *,
    graph_pair_store: GraphPairStore | None = None,
    retention: RetentionPolicy | None = None,
    accumulating: bool = False,
    gradient_dtype: torch.dtype | None = None,
    round_accumulation_once: bool = False,
) -> tuple[DifferentiatedStage, ...]:
    """Bind every stage occurrence to its structural graph pairs, their
    parameter gradients produced at ``gradient_dtype`` when one is given, and
    their ``save`` variant retaining what ``retention`` says to retain."""

    store = graph_pair_store or GraphPairStore()
    # A requires_grad output can feed a detach, an index selection or a
    # custom backward returning None. Follow actual VJPs from the objective,
    # so such values never become unproduced activation-gradient inputs.
    needed: list[set[int]] = [set() for _ in partitioned.stages]
    needed[-1].add(partitioned.user_output_indices[0])
    captured: list[DifferentiatedStage] = []
    for index in reversed(range(len(partitioned.stages))):
        stage = _capture_training_stage(
            partitioned,
            index,
            roots=tuple(sorted(needed[index])),
            graph_pair_store=store,
            retention=retention,
            accumulating=accumulating,
            gradient_dtype=gradient_dtype,
            round_accumulation_once=round_accumulation_once,
        )
        captured.append(stage)
        pair = stage.graph_pairs.reference
        output = next(
            node
            for node in pair.backward.graph_module.graph.nodes
            if node.op == "output"
        )
        gradients, _ = tree_flatten(output.args[0])
        sources = stage.example.stage.input_sources
        if len(gradients) != len(sources):
            raise CaptureError("backward gradient arity differs from stage inputs")
        for gradient, source in zip(gradients, sources, strict=True):
            if (
                gradient is None
                or source is None
                or source.producer_stage_index is None
            ):
                continue
            assert source.producer_output_index is not None
            needed[source.producer_stage_index].add(source.producer_output_index)
    return tuple(reversed(captured))


def _capture_training_stage(
    partitioned: PartitionedExport,
    stage_index: int,
    *,
    roots: tuple[int, ...],
    graph_pair_store: GraphPairStore,
    retention: RetentionPolicy | None = None,
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
    if not roots:
        raise CaptureError(
            f"training {example.stage.stage_id} has no output contributing a "
            "gradient to the objective; combine it with a differentiable consumer"
        )
    if any(position not in differentiable for position in roots):
        raise CaptureError("a stage gradient targets a nondifferentiable output")
    return DifferentiatedStage(
        example=example,
        graph_pairs=graph_pair_store.resolve(
            example,
            roots,
            specialize_unit_tangents=stage_index == len(partitioned.stages) - 1,
            retention=retention,
            accumulating=accumulating,
            gradient_dtype=gradient_dtype,
            round_accumulation_once=round_accumulation_once,
        ),
    )


__all__ = ["capture_training_stages"]
