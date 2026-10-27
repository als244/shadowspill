"""PyTorch build policies and low-level runtime integration.

Applications use the phase functions exported by :mod:`shadowspill`.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from shadowspill.runtime import MemoryPool as MemoryPool
    from shadowspill.runtime import (
        RuntimeConfigurationError as RuntimeConfigurationError,
    )
    from shadowspill.runtime import TransferCapabilities as TransferCapabilities
    from shadowspill.runtime import TransferProfile as TransferProfile
    from shadowspill.runtime.failures import (
        ExecutionTaskIdentity as ExecutionTaskIdentity,
    )
    from shadowspill.runtime.failures import (
        RuntimeExecutionError as RuntimeExecutionError,
    )
    from shadowspill.runtime.failures import (
        RuntimeFailureDiagnostics as RuntimeFailureDiagnostics,
    )

    from .contracts import ObjectiveResult as ObjectiveResult
    from .contracts import TensorSpec as TensorSpec
    from .invocation import InvocationResult as InvocationResult
    from .partition import PartitionPolicy as PartitionPolicy
    from .partition import PartitionSpec as PartitionSpec
    from .runtime import Runtime as Runtime

_GROUPS = {
    "shadowspill.runtime": (
        "MemoryPool",
        "RuntimeConfigurationError",
        "TransferCapabilities",
        "TransferProfile",
    ),
    "shadowspill.runtime.failures": (
        "ExecutionTaskIdentity",
        "RuntimeExecutionError",
        "RuntimeFailureDiagnostics",
    ),
    ".contracts": ("ObjectiveResult", "TensorSpec"),
    ".invocation": ("InvocationResult",),
    ".partition": ("PartitionPolicy", "PartitionSpec"),
    ".runtime": ("Runtime",),
}
_PUBLIC = {name: module for module, names in _GROUPS.items() for name in names}


def __getattr__(name: str) -> Any:
    if name not in _PUBLIC:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_PUBLIC[name], __name__), name)
    globals()[name] = value
    return value


__all__ = list(_PUBLIC)
