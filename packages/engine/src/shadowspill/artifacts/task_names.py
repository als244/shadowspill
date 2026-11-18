"""Short names for a program's tasks, and where in the model each one is.

A compiled task is shared by every stage that needs it, so the task alone
cannot say where it is. The parts of the model a task covers are the
registered parameters and buffers its own objects reach, and a profile's are
those of every task or saved form that uses it. Progress lines, task errors,
report rows and the report's task view all place a task through this module.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .program import Program

#: A module's path and, for a numbered member of a repeated part, its number.
Part = tuple[str, int | None]

DASH, DOTS, SEPARATOR = chr(0x2013), chr(0x2026), f" {chr(0xB7)} "


@dataclass(frozen=True)
class ProfileName:
    """What a profile's tasks do, which alternative they belong to, and where.

    ``task`` is a stage task's phase, with ``(accumulating)`` on a backward
    that adds into gradients already there, or another task's operation.
    ``variant`` is a stage task's alternative and empty otherwise. ``parts``
    is every part of the model the profile's tasks and saved forms cover.
    """

    task: str
    variant: str = ""
    parts: tuple[Part, ...] = ()

    @property
    def label(self) -> str:
        """One line: ``recompute backward / layers 3, 7, 11, ... 43``, the
        pieces joined by a middle dot."""
        what = f"{self.variant} {self.task}".strip()
        return SEPARATOR.join(filter(None, [what, describe(self.parts)]))


def part(name: str) -> Part:
    """The part of a model a registered name places a value in.

    ``layers.3.attention.weight`` is in ``("layers", 3)``, the numbered member
    of a repeated part; ``embedding.weight`` is in ``("embedding", None)``, the
    module that holds it.
    """
    pieces = name.split(".")
    for index, piece in enumerate(pieces):
        if index and piece.isdecimal():
            return ".".join(pieces[:index]), int(piece)
    return ".".join(pieces[:-1]) or name, None


def describe(parts: Iterable[Sequence[Any]], *, limit: int = 4) -> str:
    """Parts in a few words, in the order first met: ``embedding, layers 1-2``.

    The numbered members of one repeated part are joined as runs, written with
    an en dash. Past ``limit`` runs a list keeps its start and its end.
    """
    numbers: dict[str, list[int]] = {}
    for path, number in parts:
        found = numbers.setdefault(path, [])
        if number is not None:
            found.append(number)
    words = []
    for path, found in numbers.items():
        if not found:
            words.append(path)
            continue
        values = sorted(set(found))
        runs: list[list[int]] = []
        for value in values:
            if runs and value == runs[-1][1] + 1:
                runs[-1][1] = value
            else:
                runs.append([value, value])
        text = [str(a) if a == b else f"{a}{DASH}{b}" for a, b in runs]
        if len(text) > limit:
            text = [*text[: limit - 1], f"{DOTS} {text[-1]}"]
        one = path.removesuffix("s") if len(values) == 1 else path
        words.append(f"{one} {', '.join(text)}")
    return ", ".join(words)


def _stored(program: Program) -> dict[str, tuple[Part, ...]]:
    """The parts of the model each object's storage holds, by object id.

    A registration names a parameter or buffer of the model and the object
    that holds it; every object of that alias group reaches it.
    """
    groups = {item.object_id: item.alias_group_id for item in program.objects}
    held: dict[str, dict[Part, None]] = {}
    for row in program.bindings.get("registrations", ()):
        held.setdefault(groups[row["object_id"]], {}).setdefault(part(row["name"]))
    return {
        object_id: tuple(held[group])
        for object_id, group in groups.items()
        if group in held
    }


def task_parts(program: Program) -> dict[str, tuple[Part, ...]]:
    """The parts of the model each task reads or mutates, by task id.

    Empty for a task that touches no registered storage, and for every task
    of a program that registers none.
    """
    stored = _stored(program)
    result = {}
    for task in program.tasks:
        covered: dict[Part, None] = {}
        for object_id in (*task.inputs, *(item.object_id for item in task.mutations)):
            covered.update(dict.fromkeys(stored.get(object_id, ())))
        result[task.task_id] = tuple(covered)
    return result


def profile_names(program: Program) -> dict[str, ProfileName]:
    """A name for each profile key of a program.

    A stage's task is named by its phase and its alternative, and placed where
    all of its stage's tasks are, so a backward that reads no parameter itself
    is still placed. Any other task is named by its operation -- its
    definition's name, or its phase when the definition has none -- and placed
    by what its own tasks cover. A saved form names profiles that no task of this
    program runs, such as the accumulating backward of a program with one
    example, so a profile has one name on every program built from its forms.

    A name is not an identity: two profiles can share one, and the profile key
    tells them apart.
    """
    stored = _stored(program)
    tasks = {task.task_id: task for task in program.tasks}
    covers = task_parts(program)
    names: dict[str, tuple[str, str]] = {}
    places: dict[str, dict[Part, None]] = {}

    def use(profile: str, task: str, variant: str, where: Iterable[Part]) -> None:
        names.setdefault(profile, (task, variant))
        places.setdefault(profile, {}).update(dict.fromkeys(where))

    for inventory in program.bindings.get("training_templates", ()):
        for forms in inventory:
            stage: dict[Part, None] = {}
            for form in forms:
                for side in ("forward", "backward"):
                    for item in form[side]["inputs"]:
                        stage.update(dict.fromkeys(stored.get(item["object_id"], ())))
            for form in forms:
                variant = form["variant"]
                adds = " (accumulating)" if form["accumulates"] else ""
                use(form["forward"]["profile_key"], "forward", variant, stage)
                use(form["backward"]["profile_key"], f"backward{adds}", variant, stage)
    grouped: set[str] = set()
    for group in program.task_alternative_groups:
        active = [
            (option.option_id, tasks[task_id])
            for option in group.options
            for task_id in option.active_task_ids
        ]
        stage = {}
        for _, task in active:
            stage.update(dict.fromkeys(covers[task.task_id]))
        for option_id, task in active:
            grouped.add(task.task_id)
            use(task.profile_id, task.phase, option_id, stage)
    operations = {
        item.profile_key: item.name for item in program.task_definitions if item.name
    }
    for task in program.tasks:
        if task.task_id not in grouped:
            operation = operations.get(task.profile_id) or task.phase
            use(task.profile_id, operation, "", covers[task.task_id])
    for item in program.task_definitions:
        use(item.profile_key, operations.get(item.profile_key) or "task", "", ())
    return {
        profile: ProfileName(task, variant, tuple(places[profile]))
        for profile, (task, variant) in names.items()
    }


def profile_labels(program: Program) -> dict[str, str]:
    """A line-sized name for each profile key of a program: its `ProfileName`
    in one line."""
    return {key: name.label for key, name in profile_names(program).items()}


__all__ = [
    "Part",
    "ProfileName",
    "describe",
    "part",
    "profile_labels",
    "profile_names",
    "task_parts",
]
