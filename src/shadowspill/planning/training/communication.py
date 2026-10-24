"""One deterministic enqueue order for framework collective tasks."""

from dataclasses import replace

from shadowspill.ir import ResourceKind

from .tasks import TrainingSequence


def order_collectives(sequence: TrainingSequence) -> None:
    previous = None
    for index, task in enumerate(sequence.tasks):
        if task.resource.kind != ResourceKind.COMMUNICATION:
            continue
        if previous is not None:
            sequence.tasks[index] = replace(
                task,
                dependencies=tuple(dict.fromkeys((*task.dependencies, previous))),
            )
        previous = task.task_id
