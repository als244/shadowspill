"""Compose partitioned stages with structural AOT graph pairs."""

from __future__ import annotations

import torch

from shadowspill.pytorch.capture.aot import TrainingObjectiveCapture

from ..partition import PartitionSpec, partition_export
from .artifacts import PartitionedTrainingCapture
from .capture import capture_training_stages
from .store import GraphPairStore


def partition_training_capture(
    capture: TrainingObjectiveCapture,
    *,
    partition: PartitionSpec = "auto",
    graph_pair_store: GraphPairStore | None = None,
    representative_root_inputs: tuple[object, ...] | None = None,
    accumulating: bool = False,
    gradient_dtype: torch.dtype | None = None,
) -> PartitionedTrainingCapture:
    """Partition and differentiate one captured objective template.

    ``accumulating`` says this capture belongs to a microbatch that adds onto
    gradients its predecessors created, so its stages need the backward form
    that does the adding. ``gradient_dtype`` is the dtype parameter gradients
    are created and accumulated at; ``None`` keeps each at its parameter's.
    """

    partitioned = partition_export(
        capture.exported,
        capture.capture_module,
        partition=partition,
        representative_root_inputs=representative_root_inputs,
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
            accumulating=accumulating,
            gradient_dtype=gradient_dtype,
        ),
    )


__all__ = [
    "partition_training_capture",
]
