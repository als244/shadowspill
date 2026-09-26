"""Partitioned training lowering built from shared lowering primitives."""

from shadowspill.task.entrypoints import TaskEntrypoint

from .artifacts import (
    FixedTensorBinding,
    GradientBinding,
    LoweredTrainingProgram,
    OptimizerObjectBinding,
    TrainingStorageLayout,
)
from .objects import lower_training_storage_layout
from .program import lower_partitioned_training_program
from .tasks import optimizer_object_ids

__all__ = [
    "FixedTensorBinding",
    "GradientBinding",
    "LoweredTrainingProgram",
    "OptimizerObjectBinding",
    "TaskEntrypoint",
    "TrainingStorageLayout",
    "lower_partitioned_training_program",
    "lower_training_storage_layout",
    "optimizer_object_ids",
]
