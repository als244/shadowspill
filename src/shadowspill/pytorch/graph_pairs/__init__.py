"""Stage-local AOT forward/backward graph-pair artifacts and persistence."""

from .artifacts import (
    DifferentiatedStage,
    GraphPairVariant,
    PartitionedTrainingCapture,
    TaskGraphPairs,
    parameter_gradient_leaves,
)
from .capture import capture_training_stages
from .footprint import SavedValueFootprint, saved_value_footprint
from .saved_values import resolve_partitioned_saved_values
from .store import GraphPairStore
from .training import partition_training_capture

__all__ = [
    "DifferentiatedStage",
    "GraphPairStore",
    "GraphPairVariant",
    "PartitionedTrainingCapture",
    "SavedValueFootprint",
    "TaskGraphPairs",
    "capture_training_stages",
    "parameter_gradient_leaves",
    "partition_training_capture",
    "resolve_partitioned_saved_values",
    "saved_value_footprint",
]
