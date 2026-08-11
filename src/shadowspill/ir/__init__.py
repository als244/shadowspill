"""Framework-neutral logical programs and resolved execution plans."""

from ._validation import ValidationError
from .dense import (
    DenseExecutionPlan,
    DenseMemorySchedule,
    DenseProgram,
    project_dense,
    project_dense_execution_plan,
    project_dense_schedule,
)
from .execution import (
    EntrypointSpec,
    ExecutionPlan,
    PhysicalAdmission,
    PlanPrediction,
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

__all__ = [
    "AliasGroupSpec",
    "DenseExecutionPlan",
    "DenseMemorySchedule",
    "DenseProgram",
    "DeviceSpec",
    "EntrypointSpec",
    "ExecutionPlan",
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
    "TaskProfile",
    "TaskSpec",
    "ValidationError",
    "project_dense",
    "project_dense_execution_plan",
    "project_dense_schedule",
]
