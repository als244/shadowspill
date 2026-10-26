"""Optimizer capture, stage ownership, state, and task artifacts.

`capture` is the entry: it discovers the state an optimizer keeps
(`discovery`), traces its update once (`trace`) and partitions it
into tasks (`tasks`), over copies of the optimizer (`sandbox`) and the named
tensors it touches (`bindings`). `artifacts` holds their results and `store`
the trace's cache;
`staging` which stage owns each parameter, `checkpoint` the state's structure.
"""

from .artifacts import (
    OptimizerCapture,
    OptimizerTask,
    OptimizerTensorBinding,
    OptimizerTensorRole,
)
from .capture import capture_optimizer, current_optimizer_bindings
from .checkpoint import restore_optimizer_checkpoint_structure
from .staging import (
    training_parameter_stage_owners,
    training_parameters_with_gradients,
)

__all__ = [
    "OptimizerCapture",
    "OptimizerTask",
    "OptimizerTensorBinding",
    "OptimizerTensorRole",
    "capture_optimizer",
    "current_optimizer_bindings",
    "restore_optimizer_checkpoint_structure",
    "training_parameter_stage_owners",
    "training_parameters_with_gradients",
]
