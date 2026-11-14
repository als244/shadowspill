"""Stage-local AOT forward/backward graph-pair artifacts and persistence."""

from .artifacts import (
    DifferentiatedStage,
    GraphPairVariant,
    PartitionedTrainingCapture,
    TaskGraphPairs,
    parameter_gradient_leaves,
)
from .build import GRAPH_PAIR_VARIANTS, validate_graph_pair_variants
from .capture import capture_training_stages
from .footprint import SavedValueFootprint, saved_value_footprint
from .saved_values import register_saved_value_producers
from .store import GraphPairStore
from .training import partition_training_capture

__all__ = [
    "GRAPH_PAIR_VARIANTS",
    "DifferentiatedStage",
    "GraphPairStore",
    "GraphPairVariant",
    "PartitionedTrainingCapture",
    "SavedValueFootprint",
    "TaskGraphPairs",
    "capture_training_stages",
    "parameter_gradient_leaves",
    "partition_training_capture",
    "register_saved_value_producers",
    "saved_value_footprint",
    "validate_graph_pair_variants",
]
