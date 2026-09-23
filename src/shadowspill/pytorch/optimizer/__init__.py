"""Optimizer capture, stage ownership, state, and task artifacts.

`capture` is the entry: it discovers the lazy state an optimizer creates
(`discovery`), traces the recurrent update once (`trace`) and partitions it
into tasks (`tasks`), over copies of the optimizer (`sandbox`) and the named
tensors it touches (`bindings`); `opaque` materialises the bounded fallback for
profiling. `artifacts` are the values these produce, `store` the trace's cache,
`staging` which stage owns each parameter, `checkpoint` the state's structure.
"""

from .artifacts import (
    OpaqueOptimizerArtifact,
    OptimizerCapture,
    OptimizerTask,
    OptimizerTaskArtifact,
    OptimizerTensorBinding,
    OptimizerTensorRole,
)
from .capture import capture_optimizer, current_optimizer_bindings
from .checkpoint import restore_optimizer_checkpoint_structure
from .opaque import materialize_opaque_optimizer, opaque_optimizer_outputs
from .staging import (
    training_parameter_stage_owners,
    training_parameters_with_gradients,
)

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
    "training_parameters_with_gradients",
]
