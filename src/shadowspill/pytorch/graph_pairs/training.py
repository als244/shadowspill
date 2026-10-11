"""Compose partitioned stages with structural AOT graph pairs."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from shadowspill.pytorch.capture.aot import TrainingObjectiveCapture
from shadowspill.pytorch.capture.retention import RetentionPolicy

from ..partition import PartitionSpec, partition_export
from ..partition.values import StageOutputKey
from .artifacts import PartitionedTrainingCapture
from .build import GRAPH_PAIR_VARIANTS
from .capture import capture_training_stages
from .store import GraphPairStore


def partition_training_capture(
    capture: TrainingObjectiveCapture,
    *,
    partition: PartitionSpec = "auto",
    graph_pair_store: GraphPairStore | None = None,
    representative_root_inputs: tuple[object, ...] | None = None,
    representative_stage_outputs: Mapping[StageOutputKey, torch.Tensor] | None = None,
    retention: RetentionPolicy | None = None,
    accumulating: bool = False,
    gradient_dtype: torch.dtype | None = None,
    round_accumulation_once: bool = False,
    graph_pair_variants: tuple[str, ...] = GRAPH_PAIR_VARIANTS,
) -> PartitionedTrainingCapture:
    """Partition and differentiate one captured objective template.

    ``representative_stage_outputs`` hands the partition authentic control
    values already derived for this exported program, as
    :func:`partition_export` takes them; without them it derives its own.
    ``retention`` says what each stage's ``save`` variant retains for its
    backward and what it regenerates; ``None`` is the library's default.
    ``accumulating`` says this capture belongs to a microbatch that adds onto
    gradients its predecessors created, so its stages need the backward form
    that does the adding. ``gradient_dtype`` is the dtype parameter gradients
    are created and accumulated at; ``None`` keeps each at its parameter's.
    ``round_accumulation_once`` lets a matrix multiply add its product into a
    running gradient narrower than it sums at, with BLAS-controlled rounding
    (:func:`~shadowspill.pytorch.capture.aot.accumulate_gradient_outputs`).
    ``graph_pair_variants`` names the variants to build, both by default.
    """

    partitioned = partition_export(
        capture.exported,
        capture.capture_module,
        partition=partition,
        representative_root_inputs=representative_root_inputs,
        representative_stage_outputs=representative_stage_outputs,
        # Training differentiates a stage through its outputs, so a stage
        # that produces only control values belongs to the stage consuming
        # it rather than standing alone.
        fold_control_only_stages=True,
    )
    return PartitionedTrainingCapture(
        training=capture,
        partitioned=partitioned,
        stages=capture_training_stages(
            partitioned,
            graph_pair_store=graph_pair_store,
            retention=retention,
            accumulating=accumulating,
            gradient_dtype=gradient_dtype,
            round_accumulation_once=round_accumulation_once,
            variants=graph_pair_variants,
        ),
    )


__all__ = [
    "partition_training_capture",
]
