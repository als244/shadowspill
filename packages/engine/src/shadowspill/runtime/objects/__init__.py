"""Runtime-owned logical objects: the reference, and how one is made.

`references` is the value a caller holds -- an `ObjectRef`, independent of
residency. `registration` is what a runtime does with them: naming, registering,
referencing and releasing them.
Mirrors `csrc/src/runtime/objects/`.
"""

from .references import ObjectConsistency, ObjectRef
from .registration import (
    acquire_object_reference,
    register_object,
    release_object_generation,
    release_object_reference,
    reserve_pool_range,
    reserve_runtime_object_ids,
)

__all__ = [
    "ObjectConsistency",
    "ObjectRef",
    "acquire_object_reference",
    "register_object",
    "release_object_generation",
    "release_object_reference",
    "reserve_pool_range",
    "reserve_runtime_object_ids",
]
