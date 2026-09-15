"""One runtime in this process: opening it, driving it, and closing it.

This package is framework-neutral. `bootstrap` installs a runtime over a
frontend's process allocator; `core` is the `Runtime` itself, owning the pools,
routes, counters and the close; `configuration` checks what a caller asks of it;
`plan` moves one plan through its life on the runtime; `objects` names,
registers and references runtime objects; `occupancy` and `residue` read what a
pool holds and what a closing plan left behind; `calibration` measures the
transfer routes; `teardown` latches what a failed call left and makes cleanup
safe; `admission_replay` checks a planned allocation script against the exact
policy the real pool applies. `abi` is the ctypes projection all of them call
through. What a plan needs *physically* is decided before any runtime exists, so
that policy is `shadowspill.planner.admission.physical`.

A framework appears only as a :mod:`shadowspill.frontend` protocol the caller
passes to `Runtime`. Everything outside this package reaches a runtime through
the names below.
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
