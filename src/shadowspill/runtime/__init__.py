"""Low-level runtime ownership, pool configuration, diagnostics, and C bindings.

Executor creates one runtime per fresh worker and binds saved plans through
``plan/``. The runtime stays framework-neutral; a RuntimeFrontend supplies the
allocator/storage integration. Offline physical planning belongs to ``planner``.
"""

from .admission_replay import (
    AdmissionReplayDecision,
    AdmissionReplayLeaseState,
    AdmissionReplayOperation,
    AdmissionReplayOperationKind,
    AdmissionReplayResult,
    AdmissionReuseDependency,
    run_admission_replay,
)
from .configuration import RuntimeConfigurationError
from .core import Runtime
from .failures import (
    ExecutionTaskIdentity,
    RuntimeExecutionError,
    RuntimeFailureDiagnostics,
)
from .objects import ObjectConsistency, ObjectRef
from .topology import (
    MemoryPool,
    RuntimeRoute,
    TransferCapabilities,
    TransferProfile,
)

__all__ = [
    "AdmissionReplayDecision",
    "AdmissionReplayLeaseState",
    "AdmissionReplayOperation",
    "AdmissionReplayOperationKind",
    "AdmissionReplayResult",
    "AdmissionReuseDependency",
    "ExecutionTaskIdentity",
    "MemoryPool",
    "ObjectConsistency",
    "ObjectRef",
    "Runtime",
    "RuntimeConfigurationError",
    "RuntimeExecutionError",
    "RuntimeFailureDiagnostics",
    "RuntimeRoute",
    "TransferCapabilities",
    "TransferProfile",
    "run_admission_replay",
]
