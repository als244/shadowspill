"""Restore framework-neutral call bindings from portable Program metadata."""

from collections.abc import Mapping
from typing import Any

from shadowspill.task.entrypoints import TaskEntrypoint, TaskOptions
from shadowspill.task.slots import ObjectSlot, TaskStorageHandoff


def entrypoint(value: Mapping[str, Any]) -> TaskEntrypoint:
    options = dict(value.get("options", {}))
    options["contribution_slots"] = tuple(
        ObjectSlot(**dict(item)) for item in options.get("contribution_slots", ())
    )
    for name in ("public_output_leaves", "named_inputs"):
        if name in options:
            options[name] = tuple(options[name])
    return TaskEntrypoint(
        value["task_id"],
        tuple(ObjectSlot(**dict(item)) for item in value["input_slots"]),
        tuple(ObjectSlot(**dict(item)) for item in value["output_slots"]),
        tuple(value.get("replacement_output_leaves", ())),
        tuple(
            TaskStorageHandoff(**dict(item))
            for item in value.get("storage_handoffs", ())
        ),
        TaskOptions(**options),
    )
