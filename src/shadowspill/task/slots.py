"""Where a task's objects sit in its contract, and the leases it hands over.

A task's inputs and outputs are a flat sequence; an `ObjectSlot` says which of
the program's objects is at which position in it. A `TaskStorageHandoff` says
that the compiler returned one of the task's own input leases as a logically
distinct output, which is legal only where the schedule releases the source at
the same boundary.

Both are positions and identities, so neither needs a framework to express.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ObjectSlot:
    """Where one of a task's objects sits in its flattened contract."""

    leaf_index: int
    object_id: str


@dataclass(frozen=True, slots=True)
class TaskStorageHandoff:
    """Transfer one task-input lease to a distinct returned logical object.

    the compiler may return an input allocation for a logically distinct output.
    The relationship is local to this invocation: it must not merge the two
    objects' alias groups globally.  A handoff is legal only when the selected
    schedule releases ``source_object_id`` at the same task boundary.
    """

    leaf_index: int
    source_object_id: str
    destination_object_id: str


__all__ = ["ObjectSlot", "TaskStorageHandoff"]
