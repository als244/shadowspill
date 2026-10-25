"""Optimizer reduction prefixes, serialized with ordinary compute tasks."""

from collections.abc import Mapping, Sequence
from typing import Any

from shadowspill.ir import MutationSpec, ResourceKind, ResourceSpec, TaskSpec

from .tasks import TrainingSequence


def optimizer_reductions(
    templates: Sequence[Mapping[str, Any]],
    dependencies: tuple[str, ...],
    *,
    device_id: str,
    sequence: TrainingSequence,
) -> list[TaskSpec]:
    """Reduce the current accumulator immediately before its local update.

    A spilled accumulator is restored as zeros plus this rank's owned shard;
    summing those disjoint contributions reconstructs the previous total once.
    Reduction output has separate storage, so spilling the input remains safe.
    These task boundaries expose costs, not independent communication streams.
    """
    tasks = []
    resource = ResourceSpec(device_id, ResourceKind.COMPUTE)
    for item in templates:
        source = item["source"]
        prefix = f"gradient.{item['name']}"
        padding = item["padding"]
        if padding is not None:
            task = TaskSpec(
                prefix + ".pad",
                resource,
                padding["profile_key"],
                dependencies=dependencies,
                inputs=(source,),
                outputs=(padding["output"],),
                phase="gradient_local",
            )
            tasks.append(task)
            sequence.entrypoints.append(
                {
                    "task_id": task.task_id,
                    "input_slots": [{"leaf_index": 0, "object_id": source}],
                    "output_slots": [{"leaf_index": 0, "object_id": padding["output"]}],
                    "replacement_output_leaves": [],
                    "storage_handoffs": [],
                    "options": {"phase": task.phase},
                }
            )
            source, dependencies = padding["output"], (*dependencies, task.task_id)
        for index, phase in enumerate(item["phases"]):
            scatter = phase["operation"] == "reduce_scatter"
            task = TaskSpec(
                f"{prefix}.{index}.{phase['operation']}",
                resource,
                phase["profile_key"],
                dependencies=dependencies,
                inputs=(source,),
                outputs=(item["stored"],) if scatter else (),
                mutations=() if scatter else (MutationSpec(source),),
                phase="gradient_communication",
            )
            tasks.append(task)
            source, dependencies = item["stored"], (*dependencies, task.task_id)
    return tasks
