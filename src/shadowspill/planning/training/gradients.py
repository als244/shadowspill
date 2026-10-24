"""Complete new local contributions before optionally adding stored gradients."""

from collections.abc import Mapping, Sequence
from typing import Any

from shadowspill.ir import MutationSpec, ResourceKind, ResourceSpec, TaskSpec

from .tasks import TrainingSequence


def complete_gradients(
    sequence: TrainingSequence,
    templates: Sequence[Mapping[str, Any]],
    *,
    accumulate: bool,
    device_id: str,
) -> TrainingSequence:
    """Insert each reduction after all local writes, independently of traversal.

    Existing reduced values are never inputs to the local backward or a SUM.
    They participate only in the final local addition after communication ends.
    """
    insertions: dict[int, list[TaskSpec]] = {}
    previous_collective = "invocation_start"
    previous_position = 0
    writes: dict[str, list[tuple[int, str]]] = {
        item["source"]: [] for item in templates
    }
    for index, task in enumerate(sequence.tasks):
        for source in (*task.outputs, *(m.object_id for m in task.mutations)):
            if source in writes:
                writes[source].append((index, task.task_id))
    ordered = sorted(
        templates,
        key=lambda item: max((i for i, _ in writes[item["source"]]), default=-1),
    )
    for item in ordered:
        source = item["source"]
        writers = writes[source]
        if not writers:
            raise ValueError(f"{item['name']}: gradient has no local producer")
        position = max(previous_position, max(index for index, _ in writers))
        dependencies = tuple(name for _, name in writers)
        tasks = insertions.setdefault(position, [])
        prefix = f"gradient.{item['name']}"
        if item["padding"] is not None:
            padding = item["padding"]
            task = _local(
                prefix + ".pad",
                padding["profile_key"],
                (source,),
                (padding["output"],),
                (),
                dependencies,
                device_id,
                sequence,
            )
            tasks.append(task)
            source, dependencies = padding["output"], (task.task_id,)
        destination = item["new"] if accumulate else item["stored"]
        for index, phase in enumerate(item["phases"]):
            scatter = phase["operation"] == "reduce_scatter"
            task = TaskSpec(
                f"{prefix}.{index}.{phase['operation']}",
                ResourceSpec(device_id, ResourceKind.COMMUNICATION),
                phase["profile_key"],
                dependencies=tuple(dict.fromkeys((*dependencies, previous_collective))),
                inputs=(source,),
                outputs=(destination,) if scatter else (),
                mutations=() if scatter else (MutationSpec(source),),
                phase="gradient_communication",
            )
            tasks.append(task)
            source = destination
            dependencies = (*dependencies, task.task_id)
            previous_collective = task.task_id
        if accumulate:
            tasks.append(
                _local(
                    prefix + ".accumulate",
                    item["accumulation_key"],
                    (item["stored"], source),
                    (),
                    (item["stored"],),
                    dependencies,
                    device_id,
                    sequence,
                )
            )
        previous_position = position
    sequence.tasks = [
        task
        for index, original in enumerate(sequence.tasks)
        for task in (original, *insertions.get(index, ()))
    ]
    return sequence


def _local(
    identifier: str,
    key: str,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    mutations: tuple[str, ...],
    dependencies: tuple[str, ...],
    device_id: str,
    sequence: TrainingSequence,
) -> TaskSpec:
    task = TaskSpec(
        identifier,
        ResourceSpec(device_id, ResourceKind.COMPUTE),
        key,
        dependencies=dependencies,
        inputs=inputs,
        outputs=outputs,
        mutations=tuple(MutationSpec(value) for value in mutations),
        phase="gradient_local",
    )
    sequence.entrypoints.append(
        {
            "task_id": identifier,
            "input_slots": [
                {"leaf_index": i, "object_id": value} for i, value in enumerate(inputs)
            ],
            "output_slots": [
                {"leaf_index": i, "object_id": value} for i, value in enumerate(outputs)
            ],
            "replacement_output_leaves": [],
            "storage_handoffs": [],
            "options": {"phase": "gradient_local"},
        }
    )
    return task
