"""One deterministic enqueue order for framework collective tasks."""

from dataclasses import replace

from shadowspill.ir import ResourceKind

from .tasks import TrainingSequence


def order_collectives(sequence: TrainingSequence) -> None:
    previous = None
    entries = {entry["task_id"] for entry in sequence.entrypoints}
    for index, task in enumerate(sequence.tasks):
        if task.resource.kind != ResourceKind.COMMUNICATION:
            continue
        if previous is not None:
            sequence.tasks[index] = replace(
                task,
                dependencies=tuple(dict.fromkeys((*task.dependencies, previous))),
            )
        if task.task_id not in entries:
            sequence.entrypoints.append(
                {
                    "task_id": task.task_id,
                    "input_slots": [
                        {"leaf_index": i, "object_id": value}
                        for i, value in enumerate(task.inputs)
                    ],
                    "output_slots": [
                        {"leaf_index": i, "object_id": value}
                        for i, value in enumerate(task.outputs)
                    ],
                    "options": {"phase": task.phase},
                }
            )
        previous = task.task_id
