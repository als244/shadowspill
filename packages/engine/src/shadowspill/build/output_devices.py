"""Resolve backend output locations during build, before measuring or planning.

Fake/compiler devices can be provisional: PyTorch SDPA declares its seed and
offset like device tensors but returns them on CPU outside CUDA graph capture.
Actual task execution in finalization supplies this missing structural fact.
"""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from typing import Any

from shadowspill.artifacts._values import plain
from shadowspill.artifacts.program import Program
from shadowspill.artifacts.tasks import stage_jobs
from shadowspill.errors import CompilationError


def resolve_output_devices(
    program: Program, devices: Mapping[str, tuple[str | None, ...]]
) -> Program:
    bindings = plain(program.bindings)
    for key, locations in devices.items():
        expected = bindings["compiler_memory"][key]["off_device_output_leaves"]
        if any(locations[index] != "cpu" for index in expected):
            raise CompilationError(f"task {key} did not return declared CPU outputs")
        if set(locations) - {None, "cpu", "cuda"}:
            raise CompilationError(
                f"task {key} returned unsupported devices {locations}"
            )
        bindings["compiler_memory"][key]["off_device_output_leaves"] = [
            index for index, location in enumerate(locations) if location == "cpu"
        ]
    objects = {item.object_id: item for item in program.objects}
    alias_devices: dict[str, str] = {}
    calls = tuple(_calls(program))
    for key, _, outputs in calls:
        observed = devices.get(key)
        if observed is not None:
            for slot in outputs:
                location = observed[slot["leaf_index"]]
                if location is None:
                    continue
                alias = objects[slot["object_id"]].alias_group_id
                if alias_devices.setdefault(alias, location) != location:
                    raise CompilationError(
                        f"storage {alias} has conflicting output devices"
                    )
    host_aliases = {
        alias for alias, location in alias_devices.items() if location == "cpu"
    }
    for key, slots, _ in calls:
        inputs = bindings["task_inputs"].get(key)
        if inputs is None:
            continue
        geometry = dict(
            zip(inputs["tensor_positions"], inputs["tensor_geometry"], strict=True)
        )
        for slot in slots:
            if objects[slot["object_id"]].alias_group_id in host_aliases:
                geometry[slot["leaf_index"]]["device_type"] = "cpu"
    return replace(
        program,
        objects=tuple(
            replace(item, offset_bytes=0, size_bytes=0)
            if item.alias_group_id in host_aliases
            else item
            for item in program.objects
        ),
        alias_groups=tuple(
            replace(item, size_bytes=0) if item.alias_group_id in host_aliases else item
            for item in program.alias_groups
        ),
        bindings=bindings,
    )


def _calls(
    program: Program,
) -> Iterator[tuple[str, Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]]:
    """Include saved forms that a later traversal or accumulation will select."""
    tasks = {task.task_id: task for task in program.tasks}
    for entry in program.bindings["entrypoints"]:
        yield (
            tasks[entry["task_id"]].profile_id,
            entry["input_slots"],
            (
                *entry["output_slots"],
                *entry.get("options", {}).get("contribution_slots", ()),
            ),
        )
    for job in stage_jobs(program, None):
        for phase in ("forward", "backward", "optimizer", "gradient"):
            if phase in job:
                component = job[phase]
                yield (
                    component["profile_key"],
                    component["inputs"],
                    (
                        *component.get("outputs", ()),
                        *component.get("contributions", ()),
                    ),
                )
