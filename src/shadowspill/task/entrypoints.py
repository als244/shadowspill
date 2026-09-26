"""One task's place in a program: how it is called, and where its objects sit.

A program's tasks are the units a plan schedules. Each has an identity, a flat
contract of input and output objects, and whatever else the program knows about
it -- which phase it belongs to, which pass over the data produced it, which of
its outputs reach the caller.

Nothing here says what kind of program it came from. A task that accumulates into
an object rather than replacing it has `contribution_slots`, whether those
contributions are gradients or anything else; a task that addresses some of its
inputs by name rather than by position has `named_inputs`, whether those names
are an optimizer's or another program's. The frontend that
built the task knows what the names mean; the plan only needs to know they exist.

The executable behind a task is the frontend's, and is not here: a frontend keeps
its own map from `task_id` to whatever it will call.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .slots import ObjectSlot, TaskStorageHandoff


@dataclass(frozen=True, slots=True)
class TaskOptions:
    """What else the program knows about one task.

    Every field is optional because a program need say none of it. A plan reads
    what is there and does not ask why it is there.
    """

    #: Which phase of the program this task belongs to, in the program's own
    #: vocabulary, or empty where the program has one phase. A plan groups and
    #: orders by it without interpreting it.
    phase: str = ""
    #: Which partition stage produced the task, if the program was partitioned.
    stage_index: int | None = None
    #: Which repetition of its stage this task is, where the program repeats a
    #: stage over different data. The frontend's word for the repetition is the
    #: frontend's; a plan only needs to tell them apart and order them.
    repetition: int | None = None
    #: Which alternative of the task was chosen, where the program offered more
    #: than one.
    variant: str | None = None
    #: What the task was captured from, for a name a person can read.
    target: str | None = None
    #: How many of the task's outputs reach the caller, and which leaves they
    #: are. The rest are consumed inside the program.
    public_output_count: int = 0
    public_output_leaves: tuple[int, ...] = ()
    #: Outputs that accumulate into an object rather than replacing it, so the
    #: object's value is the sum of what every contributing task wrote.
    contribution_slots: tuple[ObjectSlot, ...] = ()
    #: Inputs the task addresses by name rather than by position in its
    #: contract, in the order the frontend binds them.
    named_inputs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TaskEntrypoint:
    """One task, as a program names it and a plan schedules it."""

    task_id: str
    input_slots: tuple[ObjectSlot, ...]
    output_slots: tuple[ObjectSlot, ...]
    replacement_output_leaves: tuple[int, ...] = ()
    storage_handoffs: tuple[TaskStorageHandoff, ...] = ()
    options: TaskOptions = field(default_factory=TaskOptions)


__all__ = ["TaskEntrypoint", "TaskOptions"]
