"""Framework-neutral logical programs and resolved execution plans."""

from .execution import (
    EntrypointSpec,
    ExecutionPlan,
    PhysicalAdmission,
    PlanPrediction,
)
from .indexed import (
    IndexedExecutionPlan,
    IndexedMemorySchedule,
    IndexedProgram,
    index_execution_plan,
    index_memory_schedule,
    index_program,
)
from .program import (
    AliasGroupSpec,
    DeviceSpec,
    MutationSpec,
    ObjectRole,
    ObjectSpec,
    Persistence,
    Program,
    RecomputationGroup,
    RecomputationOption,
    RecomputationSelection,
    ResourceKind,
    ResourceSpec,
    SharedResidencyPolicy,
    TaskProfile,
    TaskSpec,
)
from .schedule import (
    MemoryAction,
    MemoryActionKind,
    MemoryLocation,
    MemorySchedule,
    ResidencySpec,
)
from .sharing import SharedResidencyFootprint, shared_residency_footprint
from .validation import ValidationError

__all__ = [
    "AliasGroupSpec",
    "DeviceSpec",
    "EntrypointSpec",
    "ExecutionPlan",
    "IndexedExecutionPlan",
    "IndexedMemorySchedule",
    "IndexedProgram",
    "MemoryAction",
    "MemoryActionKind",
    "MemoryLocation",
    "MemorySchedule",
    "MutationSpec",
    "ObjectRole",
    "ObjectSpec",
    "Persistence",
    "PhysicalAdmission",
    "PlanPrediction",
    "Program",
    "RecomputationGroup",
    "RecomputationOption",
    "RecomputationSelection",
    "ResidencySpec",
    "ResourceKind",
    "ResourceSpec",
    "SharedResidencyFootprint",
    "SharedResidencyPolicy",
    "TaskProfile",
    "TaskSpec",
    "ValidationError",
    "index_execution_plan",
    "index_memory_schedule",
    "index_program",
    "shared_residency_footprint",
]
