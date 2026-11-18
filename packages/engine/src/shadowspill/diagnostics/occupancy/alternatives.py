"""Every task alternative of a program, beside what a selection chose.

A step repeats its stages: identical layers share one compiled form, and each
microbatch runs every stage again. Alternatives are therefore grouped by form,
the tasks' profiles option by option, and each form is described once with the
count of occurrences that chose each option.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from statistics import median
from typing import Any

from .facts import _mutated


def _extent(object_ids: Iterable[str], objects: Mapping[str, Mapping[str, Any]]) -> int:
    """Device bytes a set of objects reaches, each alias group's spans merged.

    Objects that view one allocation share its bytes, so sizes are never added
    across objects: a task's inputs are worth the storage they cover.
    """

    spans: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for object_id in object_ids:
        item = objects[object_id]
        start = int(item.get("offset_bytes", 0))
        spans[item["alias_group_id"]].append((start, start + int(item["size_bytes"])))
    total = 0
    for intervals in spans.values():
        cursor = None
        for start, end in sorted(intervals):
            if cursor is None or start > cursor:
                total += end - start
                cursor = end
            elif end > cursor:
                total += end - cursor
                cursor = end
    return total


def _observed(values: Sequence[int]) -> dict[str, int] | None:
    if not values:
        return None
    return {
        "count": len(values),
        "median_ns": round(median(values)),
        "min_ns": min(values),
        "max_ns": max(values),
    }


def alternatives_data(
    program: Mapping[str, Any],
    selections: Iterable[Mapping[str, str]],
    *,
    positions: Mapping[str, tuple[int, int]] | None = None,
    labels: Mapping[str, str] | None = None,
    measured_ns: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, Any]:
    """What a page's task view draws.

    ``forms`` holds one entry per distinct alternative group: its label, how
    many occurrences share it, how many chose each option, and each option's
    tasks with profiled time and sizes. ``map`` places every occurrence on a
    grid when ``positions`` gives each group a ``(column, row)``, such as its
    microbatch and stage. ``single`` lists the tasks outside every group, by
    phase and profile. ``labels`` names a profile; ``measured_ns`` gives the
    durations a trace observed for a task, and is summarised over the
    occurrences that ran.

    A task's sizes are the device bytes its inputs, outputs and mutated objects
    reach. Its total is the bytes all three reach together plus its profiled
    workspace: what must be resident for the task to run. A profile without a
    workspace leaves the total unknown rather than smaller.
    """

    labels = labels or {}
    measured_ns = measured_ns or {}
    chosen = {item["group_id"]: item["option_id"] for item in selections}
    tasks = {task["task_id"]: task for task in program["tasks"]}
    objects = {item["object_id"]: item for item in program["objects"]}
    profiles = {item["profile_id"]: item for item in program.get("profiles", ())}
    alias_size = {
        group["alias_group_id"]: int(group["size_bytes"])
        for group in program["alias_groups"]
    }

    def describe(task: Mapping[str, Any]) -> dict[str, Any]:
        inputs = tuple(task.get("inputs", ()))
        outputs = tuple(task.get("outputs", ()))
        mutated = _mutated(task)
        profile = profiles.get(task["profile_id"], {})
        workspace = profile.get("workspace_bytes")
        reach = _extent({*inputs, *outputs, *mutated}, objects)
        return {
            "phase": str(task.get("phase")),
            "label": labels.get(task["profile_id"], ""),
            "runtime_ns": profile.get("runtime_ns"),
            "input_bytes": _extent(inputs, objects),
            "output_bytes": _extent(outputs, objects),
            "mutated_bytes": _extent(mutated, objects),
            "workspace_bytes": workspace,
            "total_bytes": None if workspace is None else reach + int(workspace),
        }

    forms: dict[Any, dict[str, Any]] = {}
    form_of: dict[str, int] = {}
    grouped: set[str] = set()
    option_ids: list[str] = []
    for group in program.get("task_alternative_groups", ()):
        key = tuple(
            (
                option["option_id"],
                tuple(
                    (str(tasks[task_id].get("phase")), tasks[task_id]["profile_id"])
                    for task_id in option["active_task_ids"]
                ),
            )
            for option in group["options"]
        )
        form = forms.get(key)
        described = [
            [describe(tasks[task_id]) for task_id in option["active_task_ids"]]
            for option in group["options"]
        ]
        retained = [
            sum(alias_size[alias] for alias in option["retained_alias_group_ids"])
            for option in group["options"]
        ]
        if form is None:
            form = forms[key] = {
                "index": len(forms),
                "instances": 0,
                "sizes_vary": False,
                "options": [
                    {
                        "option": option["option_id"],
                        "chosen": 0,
                        "retained_groups": len(option["retained_alias_group_ids"]),
                        "retained_bytes": kept,
                        "tasks": rows,
                        "_measured": [[] for _ in rows],
                    }
                    for option, rows, kept in zip(
                        group["options"], described, retained, strict=True
                    )
                ],
            }
        elif any(
            entry["tasks"] != rows or entry["retained_bytes"] != kept
            for entry, rows, kept in zip(
                form["options"], described, retained, strict=True
            )
        ):
            # Equal profiles normally mean equal shapes; say so when they do not.
            form["sizes_vary"] = True
        form["instances"] += 1
        form_of[group["group_id"]] = form["index"]
        for entry, option in zip(form["options"], group["options"], strict=True):
            grouped.update(option["active_task_ids"])
            if option["option_id"] not in option_ids:
                option_ids.append(option["option_id"])
            if chosen.get(group["group_id"]) != option["option_id"]:
                continue
            entry["chosen"] += 1
            for seen, task_id in zip(
                entry["_measured"], option["active_task_ids"], strict=True
            ):
                seen.extend(measured_ns.get(task_id, ()))
    listed = []
    for form in forms.values():
        for entry in form["options"]:
            for row, seen in zip(entry["tasks"], entry.pop("_measured"), strict=True):
                row["measured"] = _observed(seen)
            times = [row["runtime_ns"] for row in entry["tasks"]]
            entry["runtime_ns"] = None if None in times else sum(times)
        first = form["options"][0]["tasks"]
        # A label that only repeats the phase names nothing; the page then
        # titles the form by where it occurs.
        form["label"] = next(
            (
                row["label"]
                for row in first
                if row["label"] and row["label"] != row["phase"]
            ),
            "",
        )
        listed.append(form)

    grid = None
    if positions is not None and form_of and form_of.keys() <= positions.keys():
        columns = sorted({positions[group][0] for group in form_of})
        rows = sorted({positions[group][1] for group in form_of})
        cells: dict[tuple[int, int], tuple[str | None, int]] = {
            (positions[group][1], positions[group][0]): (chosen.get(group), index)
            for group, index in form_of.items()
        }
        grid = {
            "columns": columns,
            "rows": [
                {
                    "row": row,
                    "cells": [
                        None if (row, column) not in cells else list(cells[row, column])
                        for column in columns
                    ],
                }
                for row in rows
            ],
        }

    single: dict[tuple[str, str], dict[str, Any]] = {}
    seen_single: dict[tuple[str, str], list[int]] = defaultdict(list)
    for task in program["tasks"]:
        if task["task_id"] in grouped:
            continue
        form_key = (str(task.get("phase")), task["profile_id"])
        if form_key not in single:
            single[form_key] = {"count": 0, **describe(task)}
        single[form_key]["count"] += 1
        seen_single[form_key].extend(measured_ns.get(task["task_id"], ()))
    for form_key, row in single.items():
        row["measured"] = _observed(seen_single[form_key])
    return {
        "options": option_ids,
        "forms": listed,
        "map": grid,
        "single": list(single.values()),
    }
