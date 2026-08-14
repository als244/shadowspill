"""PyTorch integration for the framework-neutral ShadowSpill runtime."""

from .bridge import RuntimeBridge, actions_by_task
from .failures import (
    ExecutionTaskIdentity,
    RuntimeExecutionError,
    RuntimeFailureDiagnostics,
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
    "RuntimeFailureDiagnostics",
    "TransferCapabilities",
    "TransferProfile",
    "actions_by_task",
]
