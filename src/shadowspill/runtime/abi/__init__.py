"""Declarative ctypes projection of the private framework adapter ABI.

One module per kind: the ids the header defines (``constants``), what a
caller hands the adapter to start (``configuration``), what it reports
(``statistics``), what a failure latches (``failures``), what a trace emits
(``trace``), how a plan is described (``plans``), and every entry point with
its signature (``signatures``).
"""

from __future__ import annotations

from .configuration import (
    AdapterCapabilities,
    AdapterConfig,
    PhysicalAdmission,
    PhysicalMemory,
    PoolConfig,
    RouteConfig,
    TraceConfig,
    TransferCalibrationConfig,
    TransferRouteKey,
)
from .constants import (
    ADAPTER_ABI_VERSION,
    INITIAL_ACTIONS_TASK_ID,
    PROFILING_SCOPE_BASE,
    RUNTIME_OBJECT_SCOPE_ID,
)
from .failures import AdapterFailure, RuntimeFailure
from .plans import (
    FixedDependencyDescription,
    FixedLayoutDescription,
    FixedPlacementDescription,
    ObjectBinding,
    ObjectDescription,
    ObjectLocationSnapshot,
    ObjectSnapshot,
    ObjectUpdate,
    PlanDescription,
    RuntimeAction,
    TaskAllocationContractStep,
    TaskDescription,
    TaskPublicationDescription,
)
from .signatures import (
    configure_adapter_library,
    configure_runtime_library,
    runtime_library,
)
from .statistics import (
    AdapterStatistics,
    Allocation,
    BackendStatistics,
    LiveAllocation,
    MemoryPoolStatistics,
    PlanSliceRecord,
    RuntimeStatistics,
    TransferProfile,
)
from .trace import AllocationEvent, TraceEvent, TraceSummary

__all__ = [
    "ADAPTER_ABI_VERSION",
    "INITIAL_ACTIONS_TASK_ID",
    "PROFILING_SCOPE_BASE",
    "RUNTIME_OBJECT_SCOPE_ID",
    "AdapterCapabilities",
    "AdapterConfig",
    "AdapterFailure",
    "AdapterStatistics",
    "Allocation",
    "AllocationEvent",
    "BackendStatistics",
    "FixedDependencyDescription",
    "FixedLayoutDescription",
    "FixedPlacementDescription",
    "LiveAllocation",
    "MemoryPoolStatistics",
    "ObjectBinding",
    "ObjectDescription",
    "ObjectLocationSnapshot",
    "ObjectSnapshot",
    "ObjectUpdate",
    "PhysicalAdmission",
    "PhysicalMemory",
    "PlanDescription",
    "PlanSliceRecord",
    "PoolConfig",
    "RouteConfig",
    "RuntimeAction",
    "RuntimeFailure",
    "RuntimeStatistics",
    "TaskAllocationContractStep",
    "TaskDescription",
    "TaskPublicationDescription",
    "TraceConfig",
    "TraceEvent",
    "TraceSummary",
    "TransferCalibrationConfig",
    "TransferProfile",
    "TransferRouteKey",
    "configure_adapter_library",
    "configure_runtime_library",
    "runtime_library",
]
