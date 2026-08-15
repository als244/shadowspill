"""PyTorch integration for the framework-neutral ShadowSpill runtime."""

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
    TransferCapabilities,
    TransferProfile,
)

__all__ = [
    "ExecutionTaskIdentity",
    "MemoryPool",
    "PlanMemory",
    "Runtime",
    "RuntimeBridge",
    "RuntimeConfigurationError",
    "RuntimeExecutionError",
    "RuntimeFixedDependency",
    "RuntimeFixedLayout",
    "RuntimeFixedPlacement",
    "RuntimeFailureDiagnostics",
    "RuntimePlacementKind",
    "TransferCapabilities",
    "TransferProfile",
    "actions_by_task",
]
