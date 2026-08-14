"""Framework-neutral physical admission helpers."""

from .admission import (
    AdmissionError,
    AdmissionPolicy,
    AllocationEvent,
    AllocationOperation,
    SlabLayout,
    SlabPlacement,
    SlabReplay,
    admit_physical_budget,
    plan_slab_layout,
    replay_slab_timeline,
    workspace_reserve_bytes,
)
from .memory_replay import (
    MemoryReplay,
    MemoryReplayDecision,
    MemoryReplayLeaseState,
    MemoryReplayOperation,
    MemoryReplayOperationKind,
    MemoryReuseDependency,
    replay_memory_pool,
)

__all__ = [
    "AdmissionError",
    "AdmissionPolicy",
    "AllocationEvent",
    "AllocationOperation",
    "MemoryReplay",
    "MemoryReplayDecision",
    "MemoryReplayLeaseState",
    "MemoryReplayOperation",
    "MemoryReplayOperationKind",
    "MemoryReuseDependency",
    "SlabLayout",
    "SlabPlacement",
    "SlabReplay",
    "admit_physical_budget",
    "plan_slab_layout",
    "replay_memory_pool",
    "replay_slab_timeline",
    "workspace_reserve_bytes",
]
