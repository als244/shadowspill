"""Compose partitioned stages with structural AOT graph pairs."""

from __future__ import annotations

from shadowspill.pytorch.capture.aot import TrainingObjectiveCapture

from ..partition import PartitionSpec, partition_export
from .artifacts import PartitionedTrainingCapture
from .capture import capture_training_stages
from .store import GraphPairStore


def partition_training_capture(
    capture: TrainingObjectiveCapture,
    *,
    partition: PartitionSpec = "auto",
    graph_pair_repository: GraphPairStore | None = None,
    representative_root_inputs: tuple[object, ...] | None = None,
) -> PartitionedTrainingCapture:
    """Partition and differentiate one captured objective template."""

    partitioned = partition_export(
        capture.exported,
        capture.capture_module,
        partition=partition,
        representative_root_inputs=representative_root_inputs,
    )
    return PartitionedTrainingCapture(
        training=capture,
        partitioned=partitioned,
        stages=capture_training_stages(
            partitioned, graph_pair_repository=graph_pair_repository
        ),
    )


__all__ = [
    "partition_training_capture",
]
