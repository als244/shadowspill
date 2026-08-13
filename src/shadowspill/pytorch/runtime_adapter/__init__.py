"""PyTorch integration for the framework-neutral ShadowSpill runtime."""

from .bridge import RuntimeBridge, RuntimeExecutionError, actions_by_task
from .runtime import (
    MemoryPool,
    PlanMemory,
    Runtime,
    RuntimeConfigurationError,
    TransferCapabilities,
    TransferProfile,
)

__all__ = [
    "MemoryPool",
    "PlanMemory",
    "Runtime",
    "RuntimeBridge",
    "RuntimeConfigurationError",
    "RuntimeExecutionError",
    "TransferCapabilities",
    "TransferProfile",
    "actions_by_task",
]
