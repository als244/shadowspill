"""Stage-local AOT forward/backward graph-pair artifacts and persistence."""

from .artifacts import (
    DifferentiatedStage,
    GraphPairPortfolio,
    GraphPairVariant,
    PartitionedTrainingCapture,
)
from .capture import capture_training_stages
from .controls import resolve_partitioned_saved_controls
from .footprint import SavedValueFootprint, saved_value_footprint
from .repository import GraphPairRepository
from .training import partition_training_capture

__all__ = [
    "DifferentiatedStage",
    "GraphPairPortfolio",
    "GraphPairRepository",
    "GraphPairVariant",
    "PartitionedTrainingCapture",
    "SavedValueFootprint",
    "capture_training_stages",
    "partition_training_capture",
    "resolve_partitioned_saved_controls",
    "saved_value_footprint",
]
