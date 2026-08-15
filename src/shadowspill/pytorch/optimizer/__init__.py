"""Optimizer capture, stage ownership, state, and task artifacts."""

from .artifacts import (
    OpaqueOptimizerArtifact,
    OptimizerCapture,
    OptimizerTask,
    OptimizerTaskArtifact,
    OptimizerTensorBinding,
    OptimizerTensorRole,
)
from .capture import (
    capture_optimizer,
    current_optimizer_bindings,
    materialize_opaque_optimizer,
    opaque_optimizer_outputs,
)
from .checkpoint import restore_optimizer_checkpoint_structure
from .staging import training_parameter_stage_owners

__all__ = [
    "OpaqueOptimizerArtifact",
    "OptimizerCapture",
    "OptimizerTask",
    "OptimizerTaskArtifact",
    "OptimizerTensorBinding",
    "OptimizerTensorRole",
    "capture_optimizer",
    "current_optimizer_bindings",
    "materialize_opaque_optimizer",
    "opaque_optimizer_outputs",
    "restore_optimizer_checkpoint_structure",
    "training_parameter_stage_owners",
]
