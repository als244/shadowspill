"""Runtime-owned logical objects: the reference, and how one is made.

`references` is the value a caller holds -- an `ObjectRef`, independent of
residency. `registration` is what a runtime does with them: naming, registering,
referencing and releasing them, and counting the persistent state among them.
Mirrors `csrc/src/runtime/objects/`.
"""

from .references import ObjectConsistency, ObjectRef
from .registration import (
    acquire_object_reference,
    register_object,
    release_object_generation,
    release_object_reference,
    release_persistent_state,
    require_state_operation_allowed,
    reserve_persistent_object_ids,
    reserve_runtime_object_ids,
    retain_persistent_state,
)

__all__ = [
    "ObjectConsistency",
    "ObjectRef",
    "acquire_object_reference",
    "register_object",
    "release_object_generation",
    "release_object_reference",
    "release_persistent_state",
    "require_state_operation_allowed",
    "reserve_persistent_object_ids",
    "reserve_runtime_object_ids",
    "retain_persistent_state",
]
