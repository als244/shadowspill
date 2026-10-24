"""Planning-time costs for explicit framework communication tasks in Program."""

from collections.abc import Mapping
from typing import TYPE_CHECKING

from shadowspill.ir import ObjectSpec, ResourceKind, TaskSpec

from .collectives import CollectiveCost, collective_cost
from .requests import CollectiveRequest

if TYPE_CHECKING:
    from shadowspill.artifacts.program import Program

    from .info import CommunicationInfo


def task_collectives(program: "Program") -> dict[str, tuple[CollectiveRequest, ...]]:
    """Bindings refer to task forms, independent of selected traversal/task IDs.

    Each form describes the same operation/geometry across a family of groups.
    Its memory requirements remain in TaskDefinition/Profiles; only its runtime
    comes from CommunicationInfo at planning. Opaque communication inside a
    model/provider task remains part of that task's ordinary measurement.
    """
    result = {}
    definitions = {item.profile_key for item in program.task_definitions}
    tasks_by_form: dict[str, list[TaskSpec]] = {}
    for task in program.tasks:
        tasks_by_form.setdefault(task.profile_id, []).append(task)
    objects = {item.object_id: item for item in program.objects}
    for key, description in program.bindings.get("collective_tasks", {}).items():
        if key not in definitions:
            raise ValueError(f"collective task names an unknown form: {key}")
        tasks = tasks_by_form.get(key, ())
        if not tasks or any(
            task.resource.kind != ResourceKind.COMMUNICATION for task in tasks
        ):
            raise ValueError(
                f"{key}: collective form must use the communication resource"
            )
        requests = tuple(
            CollectiveRequest(
                description["operation"],
                tuple(group),
                description["tensor_bytes"],
                description["dtype"],
            )
            for group in description["groups"]
        )
        if not requests or len({len(request.ranks) for request in requests}) != 1:
            raise ValueError(
                f"{key}: collective group family must have one local shape"
            )
        ranks = [rank for request in requests for rank in request.ranks]
        world = 1 if program.parallelism is None else program.parallelism.world_size
        if sorted(ranks) != list(range(world)):
            raise ValueError(
                f"{key}: collective groups must cover logical ranks exactly once"
            )
        for task in tasks:
            _validate_buffers(
                objects, task, requests[0], inplace=description.get("inplace", False)
            )
        result[key] = requests
    return result


def _validate_buffers(
    objects: Mapping[str, ObjectSpec],
    task: TaskSpec,
    request: CollectiveRequest,
    *,
    inplace: bool = False,
) -> None:
    if inplace:
        _validate_inplace_gather(objects, task, request)
        return
    if len(task.inputs) != 1:
        raise ValueError(f"{task.task_id}: collective requires one packed input tensor")
    source = objects[task.inputs[0]]
    if request.operation == "all_reduce":
        if (
            task.outputs
            or tuple(item.object_id for item in task.mutations) != task.inputs
        ):
            raise ValueError(f"{task.task_id}: all-reduce must mutate its input")
        input_bytes = request.tensor_bytes
    else:
        if len(task.outputs) != 1 or task.mutations:
            raise ValueError(
                f"{task.task_id}: gather/scatter requires one output tensor"
            )
        output = objects[task.outputs[0]]
        if source.alias_group_id == output.alias_group_id:
            raise ValueError(
                f"{task.task_id}: gather/scatter needs separate buffer storage"
            )
        gathered = request.operation == "all_gather"
        local = request.tensor_bytes // len(request.ranks)
        input_bytes = local if gathered else request.tensor_bytes
        output_bytes = request.tensor_bytes if gathered else local
        if output.size_bytes != output_bytes:
            raise ValueError(
                f"{task.task_id}: collective output bytes disagree with geometry"
            )
    if source.size_bytes != input_bytes:
        raise ValueError(
            f"{task.task_id}: collective input bytes disagree with geometry"
        )


def _validate_inplace_gather(
    objects: Mapping[str, ObjectSpec], task: TaskSpec, request: CollectiveRequest
) -> None:
    if (
        request.operation != "all_gather"
        or len(task.inputs) != 2
        or task.outputs
        or tuple(item.object_id for item in task.mutations) != task.inputs[1:]
    ):
        raise ValueError(
            f"{task.task_id}: in-place gather needs an owned view "
            "and mutable full buffer"
        )
    source, target = (objects[value] for value in task.inputs)
    chunk = request.tensor_bytes // len(request.ranks)
    if (
        source.alias_group_id != target.alias_group_id
        or target.offset_bytes != 0
        or source.size_bytes != chunk
        or target.size_bytes != request.tensor_bytes
        or source.offset_bytes % chunk != 0
        or source.offset_bytes + chunk > target.size_bytes
    ):
        raise ValueError(
            f"{task.task_id}: in-place gather views disagree with padded shard geometry"
        )


def task_collective_cost(
    requests: tuple[CollectiveRequest, ...],
    info: "CommunicationInfo",
    setup: Mapping[str, object],
) -> CollectiveCost:
    """One symmetric schedule uses the slowest group instance's completion."""
    return max(
        (
            collective_cost(info, **request.to_dict(), setup=setup)
            for request in requests
        ),
        key=lambda cost: cost.duration_seconds,
    )
