"""Lazy legacy exports while callers migrate to the phase APIs."""

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

    from .api import build_step_programs as build_step_programs
    from .api import plan_forward as plan_forward
    from .api import plan_step as plan_step
    from .api import prepare_step as prepare_step
    from .callables import PlannedForward as PlannedForward
    from .callables import PlannedTrainStep as PlannedTrainStep
    from .contracts import ObjectiveResult as ObjectiveResult
    from .contracts import TensorSpec as TensorSpec
    from .diagnostics import DiagnosticsHandle as DiagnosticsHandle
    from .diagnostics import InvocationTiming as InvocationTiming
    from .diagnostics import LaneSummary as LaneSummary
    from .diagnostics import PhaseTimingComparison as PhaseTimingComparison
    from .diagnostics import PlanAllocationEvent as PlanAllocationEvent
    from .diagnostics import PlanCacheArtifact as PlanCacheArtifact
    from .diagnostics import PlanCompiledOutputView as PlanCompiledOutputView
    from .diagnostics import PlanCompiledRoot as PlanCompiledRoot
    from .diagnostics import PlanCompilerProfile as PlanCompilerProfile
    from .diagnostics import PlanDiagnostics as PlanDiagnostics
    from .diagnostics import PlanGraphPair as PlanGraphPair
    from .diagnostics import PlanGraphProfile as PlanGraphProfile
    from .diagnostics import PlanMutationBinding as PlanMutationBinding
    from .diagnostics import PlanObjectFootprint as PlanObjectFootprint
    from .diagnostics import PlanOutputView as PlanOutputView
    from .diagnostics import PlanPhaseTiming as PlanPhaseTiming
    from .diagnostics import PlanProfilingMetadata as PlanProfilingMetadata
    from .diagnostics import PlanReport as PlanReport
    from .diagnostics import PlanStorageRoot as PlanStorageRoot
    from .diagnostics import PlanTaskStage as PlanTaskStage
    from .diagnostics import PlanUniqueStage as PlanUniqueStage
    from .diagnostics import StepDiagnostics as StepDiagnostics
    from .diagnostics import StepResult as StepResult
    from .diagnostics import StepTimingSummary as StepTimingSummary
    from .diagnostics import TaskRecord as TaskRecord
    from .diagnostics import Timelines as Timelines
    from .diagnostics import TransferQueue as TransferQueue
    from .diagnostics import TransferRecord as TransferRecord
    from .diagnostics import TransferRecords as TransferRecords
    from .distributed import Distributed as Distributed
    from .distributed import TransferSharding as TransferSharding
    from .invocation import InvocationResult as InvocationResult
    from .partition import PartitionPolicy as PartitionPolicy
    from .partition import PartitionSpec as PartitionSpec
    from .prepared import PreparedStep as PreparedStep
    from .prepared_api import admit_step as admit_step
    from .prepared_api import search_prepared_steps as search_prepared_steps
    from .profiling import ProfilingOptions as ProfilingOptions
    from .runtime import Runtime as Runtime
    from .sharing import ObjectConsistency as ObjectConsistency
    from .sharing import SharedInput as SharedInput
    from .sharing import SharedOutput as SharedOutput
    from .sharing import TensorRef as TensorRef
    from .sharing import shared_input as shared_input
    from .sharing import shared_output as shared_output
    from .state import PoolImage as PoolImage
    from .state import export_model_state as export_model_state
    from .state import export_optimizer_state as export_optimizer_state
    from .state import import_model_state as import_model_state
    from .state import import_model_state_from_file as import_model_state_from_file
    from .state import import_optimizer_state as import_optimizer_state
    from .state import (
        import_optimizer_state_from_file as import_optimizer_state_from_file,
    )
    from .state import initialize_storage_ as initialize_storage_
    from .state import read_model_state as read_model_state
    from .state import read_optimizer_state as read_optimizer_state
    from .state import release_model_state as release_model_state
    from .state import write_pool_image as write_pool_image
    from .step_search import GraphPairOutcome as GraphPairOutcome
    from .step_search import StepSearchGeometryBuild as StepSearchGeometryBuild
    from .step_search import StepSearchPoint as StepSearchPoint
    from .step_search import StepSearchReport as StepSearchReport
    from .step_search import plan_step_search as plan_step_search

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
    ".api": ("build_step_programs", "plan_forward", "plan_step", "prepare_step"),
    ".callables": ("PlannedForward", "PlannedTrainStep"),
    ".contracts": ("ObjectiveResult", "TensorSpec"),
    ".diagnostics": (
        "DiagnosticsHandle",
        "InvocationTiming",
        "LaneSummary",
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
        "StepDiagnostics",
        "StepResult",
        "StepTimingSummary",
        "TaskRecord",
        "Timelines",
        "TransferQueue",
        "TransferRecord",
        "TransferRecords",
    ),
    ".invocation": ("InvocationResult",),
    ".partition": ("PartitionPolicy", "PartitionSpec"),
    ".prepared": ("PreparedStep",),
    ".prepared_api": ("admit_step", "search_prepared_steps"),
    ".profiling": ("ProfilingOptions",),
    ".runtime": ("Runtime",),
    ".sharing": (
        "ObjectConsistency",
        "SharedInput",
        "SharedOutput",
        "TensorRef",
        "shared_input",
        "shared_output",
    ),
    ".state": (
        "PoolImage",
        "export_model_state",
        "export_optimizer_state",
        "import_model_state",
        "import_model_state_from_file",
        "import_optimizer_state",
        "import_optimizer_state_from_file",
        "initialize_storage_",
        "read_model_state",
        "read_optimizer_state",
        "release_model_state",
        "write_pool_image",
    ),
    ".step_search": (
        "GraphPairOutcome",
        "StepSearchGeometryBuild",
        "StepSearchPoint",
        "StepSearchReport",
        "plan_step_search",
    ),
    ".distributed": ("Distributed", "TransferSharding"),
}
_PUBLIC = {name: module for module, names in _GROUPS.items() for name in names}


def __getattr__(name: str) -> Any:
    if name not in _PUBLIC:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_PUBLIC[name], __name__), name)
    globals()[name] = value
    return value


__all__ = list(_PUBLIC)
