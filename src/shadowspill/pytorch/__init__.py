"""Public PyTorch values and planning entrypoints for ShadowSpill."""

from shadowspill.runtime import (
    MemoryPool,
    RuntimeConfigurationError,
    TransferCapabilities,
    TransferProfile,
)
from shadowspill.runtime.failures import (
    ExecutionTaskIdentity,
    RuntimeExecutionError,
    RuntimeFailureDiagnostics,
)

from .api import build_step_programs, plan_forward, plan_step
from .callables import PlannedForward, PlannedTrainStep
from .contracts import (
    ObjectiveResult,
    TensorSpec,
)
from .diagnostics import (
    DiagnosticsHandle,
    InvocationTiming,
    LaneSummary,
    PhaseTimingComparison,
    PlanAllocationEvent,
    PlanCacheArtifact,
    PlanCompiledOutputView,
    PlanCompiledRoot,
    PlanCompilerProfile,
    PlanDiagnostics,
    PlanGraphPair,
    PlanGraphProfile,
    PlanMutationBinding,
    PlanObjectFootprint,
    PlanOutputView,
    PlanPhaseTiming,
    PlanProfilingMetadata,
    PlanReport,
    PlanStorageRoot,
    PlanTaskStage,
    PlanUniqueStage,
    StepDiagnostics,
    StepResult,
    StepTimingSummary,
    TaskRecord,
    Timelines,
    TransferQueue,
    TransferRecord,
    TransferRecords,
)
from .invocation import InvocationResult
from .partition import PartitionPolicy, PartitionSpec
from .profiling import ProfilingOptions
from .runtime import Runtime
from .sharing import (
    ObjectConsistency,
    SharedInput,
    SharedOutput,
    TensorRef,
    shared_input,
    shared_output,
)
from .state import (
    PoolImage,
    export_model_state,
    export_optimizer_state,
    import_model_state,
    import_model_state_from_file,
    import_optimizer_state,
    import_optimizer_state_from_file,
    initialize_storage_,
    read_model_state,
    read_optimizer_state,
    release_model_state,
    write_pool_image,
)
from .step_search import (
    GraphPairOutcome,
    StepSearchGeometryBuild,
    StepSearchPoint,
    StepSearchReport,
    plan_step_search,
)

__all__ = [
    "DiagnosticsHandle",
    "ExecutionTaskIdentity",
    "GraphPairOutcome",
    "InvocationResult",
    "InvocationTiming",
    "LaneSummary",
    "MemoryPool",
    "ObjectConsistency",
    "ObjectiveResult",
    "PartitionPolicy",
    "PartitionSpec",
    "PhaseTimingComparison",
    "PlanAllocationEvent",
    "PlanCacheArtifact",
    "PlanCompiledOutputView",
    "PlanCompiledRoot",
    "PlanCompilerProfile",
    "PlanDiagnostics",
    "PlanGraphPair",
    "PlanGraphProfile",
    "PlanMutationBinding",
    "PlanObjectFootprint",
    "PlanOutputView",
    "PlanPhaseTiming",
    "PlanProfilingMetadata",
    "PlanReport",
    "PlanStorageRoot",
    "PlanTaskStage",
    "PlanUniqueStage",
    "PlannedForward",
    "PlannedTrainStep",
    "PoolImage",
    "ProfilingOptions",
    "Runtime",
    "RuntimeConfigurationError",
    "RuntimeExecutionError",
    "RuntimeFailureDiagnostics",
    "SharedInput",
    "SharedOutput",
    "StepDiagnostics",
    "StepResult",
    "StepSearchGeometryBuild",
    "StepSearchPoint",
    "StepSearchReport",
    "StepTimingSummary",
    "TaskRecord",
    "TensorRef",
    "TensorSpec",
    "Timelines",
    "TransferCapabilities",
    "TransferProfile",
    "TransferQueue",
    "TransferRecord",
    "TransferRecords",
    "build_step_programs",
    "export_model_state",
    "export_optimizer_state",
    "import_model_state",
    "import_model_state_from_file",
    "import_optimizer_state",
    "import_optimizer_state_from_file",
    "initialize_storage_",
    "plan_forward",
    "plan_step",
    "plan_step_search",
    "read_model_state",
    "read_optimizer_state",
    "release_model_state",
    "shared_input",
    "shared_output",
    "write_pool_image",
]

from .distributed import Distributed as Distributed
from .distributed import TransferSharding as TransferSharding

__all__.append("Distributed")
__all__.append("TransferSharding")
