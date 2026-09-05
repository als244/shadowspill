"""PyTorch integration for the framework-neutral ShadowSpill runtime."""

from .abi import INITIAL_ACTIONS_TASK_ID
from .bridge import RuntimeBridge, actions_by_task
from .failures import (
    ExecutionTaskIdentity,
    RuntimeExecutionError,
    RuntimeFailureDiagnostics,
)
from .fixed_layout import (
    RuntimeFixedDependency,
    RuntimeFixedLayout,
    RuntimeFixedPlacement,
    RuntimePlacementKind,
)
from .runtime import (
    MemoryPool,
    PlanMemory,
    Runtime,
    RuntimeConfigurationError,
    RuntimeRoute,
    TransferCapabilities,
    TransferProfile,
)

__all__ = [
    "INITIAL_ACTIONS_TASK_ID",
    "ExecutionTaskIdentity",
    "MemoryPool",
    "PlanMemory",
    "Runtime",
    "RuntimeBridge",
    "RuntimeConfigurationError",
    "RuntimeExecutionError",
    "RuntimeFailureDiagnostics",
    "RuntimeFixedDependency",
    "RuntimeFixedLayout",
    "RuntimeFixedPlacement",
    "RuntimePlacementKind",
    "RuntimeRoute",
    "TransferCapabilities",
    "TransferProfile",
    "actions_by_task",
]
