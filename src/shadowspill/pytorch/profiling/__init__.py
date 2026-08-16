"""Task-local profiling, representative values, metadata, and cache APIs."""

from .allocation_abi import (
    TaskAllocationABI,
    TaskAllocationABIStep,
    TaskAllocationPathObservation,
    compare_allocation_path,
)
from .allocation_core import (
    AllocationPathProbe,
    AmbiguousAllocationPathError,
    DerivedAllocationCore,
    derive_core_allocation_path,
)
from .context import ProfileInputContext, profile_input_context_digest
from .environment import profile_environment
from .inputs import (
    REPRESENTATIVE_VALUE_POLICY,
    RepresentativeInputSet,
    RepresentativeInputSummary,
    materialize_representative_inputs,
)
from .manifests import (
    ResolvedTaskManifests,
    resolve_task_manifests,
    validate_compiled_profile,
)
from .metadata import (
    ProfilingMetadata,
    canonicalize_profiling_metadata,
    training_profiling_metadata,
)
from .records import (
    PROFILE_SCHEMA,
    ProfileEnvironment,
    ProfileKey,
    ProfilingResult,
    TaskAllocationEvent,
    TaskAllocationOperation,
    TaskMeasurement,
    TaskOutputInputBinding,
)
from .repository import PlanningArtifactRecorder, ProfileRepository
from .runner import ProfilableArtifact, profile_unique_artifacts

__all__ = [
    "PROFILE_SCHEMA",
    "REPRESENTATIVE_VALUE_POLICY",
    "AllocationPathProbe",
    "AmbiguousAllocationPathError",
    "DerivedAllocationCore",
    "PlanningArtifactRecorder",
    "ProfilableArtifact",
    "ProfileEnvironment",
    "ProfileInputContext",
    "ProfileKey",
    "ProfileRepository",
    "ProfilingMetadata",
    "ProfilingResult",
    "RepresentativeInputSet",
    "RepresentativeInputSummary",
    "ResolvedTaskManifests",
    "TaskAllocationABI",
    "TaskAllocationABIStep",
    "TaskAllocationEvent",
    "TaskAllocationOperation",
    "TaskAllocationPathObservation",
    "TaskMeasurement",
    "TaskOutputInputBinding",
    "canonicalize_profiling_metadata",
    "compare_allocation_path",
    "derive_core_allocation_path",
    "materialize_representative_inputs",
    "profile_environment",
    "profile_input_context_digest",
    "profile_unique_artifacts",
    "resolve_task_manifests",
    "training_profiling_metadata",
    "validate_compiled_profile",
]
