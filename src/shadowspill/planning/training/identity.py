"""Separate compact runtime task identities from semantic example/stage names."""

from dataclasses import replace

from .tasks import TrainingSequence


def canonicalize_tasks(sequence: TrainingSequence) -> dict[str, str]:
    names = sorted(
        (task.task_id for task in sequence.tasks),
        key=lambda name: (name != "invocation_start", name),
    )
    identifiers = {name: f"task_{index:06d}" for index, name in enumerate(names)}
    sequence.tasks = [
        replace(
            task,
            task_id=identifiers[task.task_id],
            dependencies=tuple(identifiers[item] for item in task.dependencies),
        )
        for task in sequence.tasks
    ]
    sequence.groups = [
        replace(
            group,
            options=tuple(
                replace(
                    option,
                    active_task_ids=tuple(
                        identifiers[item] for item in option.active_task_ids
                    ),
                )
                for option in group.options
            ),
        )
        for group in sequence.groups
    ]
    sequence.entrypoints = [
        {**entry, "task_id": identifiers[entry["task_id"]]}
        for entry in sequence.entrypoints
    ]
    sequence.transient_outputs = {
        identifiers[key]: value for key, value in sequence.transient_outputs.items()
    }
    return {identifier: name for name, identifier in identifiers.items()}
