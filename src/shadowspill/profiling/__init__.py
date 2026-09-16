"""What a task cost, and whether it kept its allocation promise.

Framework-neutral: measuring a task needs a framework, but storing what was
measured, reading it back, and checking that a run's allocations matched the
contract do not. the frontend's profiling package measures; this holds the
result.

`store` is the profile store and the modes that gate it; `manifest_store` caches
one compiled manifest per profile key, for the toolchain it was told it caches
for; `invariant` is the allocation-path check a profiled run has to satisfy;
`metadata` is what a profile records about the machine it ran on; `timing` is the
clock a measurement is taken against.
"""

from .invariant import derive_invariant_allocation_path
from .manifest_store import CompiledManifestStore
from .metadata import ProfilingMetadata
from .store import ProfileStore

__all__ = [
    "CompiledManifestStore",
    "ProfileStore",
    "ProfilingMetadata",
    "derive_invariant_allocation_path",
]
