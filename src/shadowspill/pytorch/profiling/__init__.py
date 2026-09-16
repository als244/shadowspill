"""Task-local profiling, representative values, metadata, and cache APIs."""

from shadowspill.profiling.invariant import (
    AllocationPathProbe,
    AmbiguousAllocationPathError,
    DerivedAllocationInvariant,
    derive_invariant_allocation_path,
)
from shadowspill.profiling.metadata import (
    ProfilingMetadata,
    canonicalize_profiling_metadata,
    repeated_profiling_metadata,
)
from shadowspill.profiling.store import ProfileStore
from shadowspill.task.allocations import (
    TaskAllocationContract,
    TaskAllocationContractStep,
    TaskAllocationPathObservation,
    compare_allocation_path,
)
from shadowspill.task.profiles import (
    PROFILE_SCHEMA,
    ProfileEnvironment,
    ProfileKey,
    ProfilingResult,
    TaskAllocationEvent,
    TaskAllocationOperation,
    TaskMeasurement,
    TaskOutputInputBinding,
)

from .environment import profile_environment
from .inputs import (
    RepresentativeInputSet,
    materialize_representative_inputs,
)
from .manifests import (
    ResolvedTaskManifests,
    resolve_task_manifests,
    validate_compiled_profile,
)
from .runner import ProfilableArtifact, profile_unique_artifacts

__all__ = [
    "PROFILE_SCHEMA",
    "AllocationPathProbe",
    "AmbiguousAllocationPathError",
    "DerivedAllocationInvariant",
    "ProfilableArtifact",
    "ProfileEnvironment",
    "ProfileKey",
    "ProfileStore",
    "ProfilingMetadata",
    "ProfilingResult",
    "RepresentativeInputSet",
    "ResolvedTaskManifests",
    "TaskAllocationContract",
    "TaskAllocationContractStep",
    "TaskAllocationEvent",
    "TaskAllocationOperation",
    "TaskAllocationPathObservation",
    "TaskMeasurement",
    "TaskOutputInputBinding",
    "canonicalize_profiling_metadata",
    "compare_allocation_path",
    "derive_invariant_allocation_path",
    "materialize_representative_inputs",
    "profile_environment",
    "profile_unique_artifacts",
    "repeated_profiling_metadata",
    "resolve_task_manifests",
    "validate_compiled_profile",
]
