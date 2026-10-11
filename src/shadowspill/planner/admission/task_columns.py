"""The admission facts' tasks as arrays, in the facts' own order.

Two things read every task of a set of facts: the digest that names the facts,
and the compile that indexes them for the library. A large step's tasks carry
millions of allocation steps, so both read these columns, extracted once per
tuple of tasks -- which every capacity variant of the facts shares -- with one
pass per field and no Python object per step.

Alias names are held once, in `alias_ids`, in the order the tasks first name
them; every alias column holds positions in it. Optional fields hold `-1`
where a step names nothing. Every value a field can take is otherwise
non-negative, so the columns and the two name tables determine the tasks, and
their digest names the tasks exactly.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, fields
from itertools import chain, compress, repeat
from operator import attrgetter, is_, is_not

import numpy as np

from . import TaskAdmissionSpec, TaskAllocationStepKind

#: Step kinds as the library numbers them.
ALLOCATE = 0
RELEASE = 1


@dataclass(frozen=True, slots=True)
class TaskColumns:
    """Every field of a tuple of `TaskAdmissionSpec`, as columns.

    A per-task list is a pair: `<name>_offsets`, one more than there are
    tasks, and the values, task by task. Allocation steps are such a list,
    with one column per step field.
    """

    task_ids: tuple[str, ...]
    alias_ids: tuple[str, ...]
    workspace_offsets: np.ndarray
    workspace_extents: np.ndarray
    fresh_offsets: np.ndarray
    fresh_aliases: np.ndarray
    replacement_offsets: np.ndarray
    replacement_aliases: np.ndarray
    handoff_offsets: np.ndarray
    handoff_sources: np.ndarray
    handoff_destinations: np.ndarray
    step_offsets: np.ndarray
    step_ordinals: np.ndarray
    step_kinds: np.ndarray
    step_bytes: np.ndarray
    step_outputs: np.ndarray
    step_reuses: np.ndarray

    def digest(self) -> str:
        """The sha256 of the two name tables and every column, each column
        preceded by its name, element type and length."""

        hasher = hashlib.sha256()
        hasher.update(
            json.dumps([self.task_ids, self.alias_ids], separators=(",", ":")).encode()
        )
        for item in fields(self):
            values = getattr(self, item.name)
            if isinstance(values, np.ndarray):
                hasher.update(
                    f";{item.name}:{values.dtype.str}:{len(values)};".encode()
                )
                hasher.update(np.ascontiguousarray(values))
        return hasher.hexdigest()


def _ragged(
    tasks: Sequence[TaskAdmissionSpec], field: str
) -> tuple[np.ndarray, tuple[object, ...]]:
    """A per-task tuple field as offsets and its values, task by task."""

    rows = list(map(attrgetter(field), tasks))
    offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    np.cumsum(
        np.fromiter(map(len, rows), dtype=np.int64, count=len(rows)), out=offsets[1:]
    )
    return offsets, tuple(chain.from_iterable(rows))


def _integers(values: Sequence[object] | object, count: int) -> np.ndarray:
    return np.fromiter(values, dtype=np.int64, count=count)


def _optional(values: list[object]) -> tuple[np.ndarray, list[object]]:
    """Which entries are present, and the present ones in order."""

    present = np.fromiter(
        map(is_not, values, repeat(None)), dtype=np.bool_, count=len(values)
    )
    return present, list(compress(values, present.tolist()))


def task_columns(tasks: Sequence[TaskAdmissionSpec]) -> TaskColumns:
    """Extract `tasks` into columns, in their own order."""

    task_ids = tuple(map(attrgetter("task_id"), tasks))
    workspace_offsets, extents = _ragged(tasks, "workspace_extents")
    fresh_offsets, fresh = _ragged(tasks, "fresh_output_aliases")
    replacement_offsets, replacements = _ragged(tasks, "replacement_aliases")
    handoff_offsets, handoffs = _ragged(tasks, "storage_handoffs")
    sources = tuple(map(attrgetter("source_alias_group_id"), handoffs))
    destinations = tuple(map(attrgetter("destination_alias_group_id"), handoffs))
    step_offsets, steps = _ragged(tasks, "allocation_steps")
    count = len(steps)
    present_outputs, outputs = _optional(
        list(map(attrgetter("output_alias_group_id"), steps))
    )
    present_reuses, reuses = _optional(
        list(map(attrgetter("reuses_allocation_ordinal"), steps))
    )
    alias_ids = tuple(
        dict.fromkeys(chain(fresh, replacements, sources, destinations, outputs))
    )
    position = dict(zip(alias_ids, range(len(alias_ids)), strict=True))

    def positions(names: Sequence[object]) -> np.ndarray:
        return _integers(map(position.__getitem__, names), len(names))

    step_outputs = np.full(count, -1, dtype=np.int64)
    step_outputs[present_outputs] = positions(outputs)
    step_reuses = np.full(count, -1, dtype=np.int64)
    step_reuses[present_reuses] = _integers(reuses, len(reuses))
    return TaskColumns(
        task_ids=task_ids,
        alias_ids=alias_ids,
        workspace_offsets=workspace_offsets,
        workspace_extents=_integers(extents, len(extents)),
        fresh_offsets=fresh_offsets,
        fresh_aliases=positions(fresh),
        replacement_offsets=replacement_offsets,
        replacement_aliases=positions(replacements),
        handoff_offsets=handoff_offsets,
        handoff_sources=positions(sources),
        handoff_destinations=positions(destinations),
        step_offsets=step_offsets,
        step_ordinals=_integers(map(attrgetter("allocation_ordinal"), steps), count),
        step_kinds=np.fromiter(
            map(
                is_,
                map(attrgetter("kind"), steps),
                repeat(TaskAllocationStepKind.RELEASE),
            ),
            dtype=np.uint8,
            count=count,
        ),
        step_bytes=_integers(map(attrgetter("charged_bytes"), steps), count),
        step_outputs=step_outputs,
        step_reuses=step_reuses,
    )


__all__ = ["ALLOCATE", "RELEASE", "TaskColumns", "task_columns"]
