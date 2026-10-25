"""Place local updates after every required gradient write and old-weight read."""

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from shadowspill.ir import MutationSpec, ResourceKind, ResourceSpec, TaskSpec

from .gradients import optimizer_reductions
from .tasks import TrainingSequence


def place_optimizer(
    sequence: TrainingSequence,
    components: Sequence[Mapping[str, Any]],
    *,
    ordering: Literal["stage_interleaved", "tail"],
    aliases: Mapping[str, str],
    device_id: str,
) -> TrainingSequence:
    if ordering not in {"stage_interleaved", "tail"}:
        raise ValueError("optimizer_ordering must be 'stage_interleaved' or 'tail'")
    if not components:
        return sequence
    # Include reads of aliased views, not only the canonical parameter object.
    touches: dict[str, list[tuple[int, str]]] = {}
    for index, task in enumerate(sequence.tasks):
        for value in (
            *task.inputs,
            *task.outputs,
            *(mutation.object_id for mutation in task.mutations),
        ):
            touches.setdefault(aliases[value], []).append((index, task.task_id))
    insertions: dict[int, list[TaskSpec]] = {}
    entries = {entry["task_id"]: entry for entry in sequence.entrypoints}
    previous = "invocation_start"
    previous_position = 0
    for component in components:
        inputs = tuple(dict.fromkeys(slot["object_id"] for slot in component["inputs"]))
        reductions = component.get("reductions", ())
        accesses = [
            touch
            for value in (*inputs, *(item["source"] for item in reductions))
            for touch in touches.get(aliases[value], ())
        ]
        position = (
            len(sequence.tasks) - 1
            if ordering == "tail"
            else max((index for index, _ in accesses), default=0)
        )
        position = max(position, previous_position)
        dependencies = tuple(dict.fromkeys((previous, *(task for _, task in accesses))))
        block = insertions.setdefault(position, [])
        prefix = optimizer_reductions(
            reductions, dependencies, device_id=device_id, sequence=sequence
        )
        block.extend(prefix)
        if prefix:
            dependencies = (*dependencies, *(task.task_id for task in prefix))
        identifier = component["task_id"]
        task = TaskSpec(
            identifier,
            ResourceSpec(device_id, ResourceKind.COMPUTE),
            component["profile_key"],
            dependencies=dependencies,
            inputs=inputs,
            mutations=tuple(MutationSpec(value) for value in component["mutations"]),
            phase="optimizer",
        )
        block.append(task)
        previous = identifier
        for publication in component.get("publications", ()):
            publish_id = f"{identifier}.publish.{publication['name']}"
            destination = publication["destination"]
            block.append(
                TaskSpec(
                    publish_id,
                    ResourceSpec(device_id, ResourceKind.COMPUTE),
                    publication["profile_key"],
                    dependencies=(previous,),
                    inputs=(publication["source"], destination),
                    mutations=(MutationSpec(destination),),
                    phase="weight_communication",
                )
            )
            previous = publish_id
        entries[identifier] = {
            "task_id": identifier,
            "input_slots": component["inputs"],
            "host_inputs": component["host_inputs"],
            "output_slots": [],
            "replacement_output_leaves": component["replacements"],
            "storage_handoffs": [],
            "options": {"phase": "optimizer"},
        }
        sequence.transient_outputs[identifier] = tuple(component["replacements"])
        previous_position = position
    # Padding entrypoints were added while assembling the optimizer prefixes.
    entries.update({entry["task_id"]: entry for entry in sequence.entrypoints})
    tasks = [
        task
        for index, original in enumerate(sequence.tasks)
        for task in (original, *insertions.get(index, ()))
    ]
    sequence.tasks = tasks
    sequence.entrypoints = [
        entries[task.task_id] for task in tasks if task.task_id in entries
    ]
    return sequence
