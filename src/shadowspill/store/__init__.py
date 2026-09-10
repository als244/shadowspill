"""Content-addressed artifact storage, and the modes that gate it.

A store is not part of planning. It holds what each stage of the pipeline
produced -- captured graphs, compiled profiles, programs, plans -- keyed by the
digest of its inputs, so a later run that asks the same question reads the
answer instead of recomputing it. The frontend's own stores, the planner's plan
store, and the qualification harnesses all sit on this one.

`StoreMode` says what a call may do with a tree, and `StorePolicy` turns one
mode into the four gates the code actually checks, so a caller never spells
out a combination that means nothing. `policy` lists the modes.
"""

from __future__ import annotations

from .artifacts import ArtifactStore, PlanningArtifact, digest_directory
from .policy import CONTRIBUTE, STORE_MODES, StoreMode, StorePolicy

__all__ = [
    "CONTRIBUTE",
    "STORE_MODES",
    "ArtifactStore",
    "PlanningArtifact",
    "StoreMode",
    "StorePolicy",
    "digest_directory",
]
