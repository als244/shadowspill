"""Find norm observation points without changing the saved Program or schedule."""

from dataclasses import dataclass
from math import prod

from shadowspill.artifacts.tensors import registration_name
from shadowspill.parallelism import GroupRef
from shadowspill.parallelism.ownership import StorageOwnership

from ..paths import ExecutionPath


@dataclass(frozen=True)
class NormInput:
    column: int
    argument: int
    elements: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    contribute: bool


def norm_inputs(
    path: ExecutionPath, logical_rank: int
) -> tuple[tuple[str, ...], dict[str, tuple[NormInput, ...]]]:
    bindings = path.plan.program.bindings
    gradients = {
        item["object_id"]: item["parameter_name"]
        for item in bindings["optimizer_bindings"]
        if item["role"] == "gradient"
    }
    names = tuple(sorted(set(gradients.values())))
    columns = {name: 2 * index for index, name in enumerate(names)}
    weights = {
        item["object_id"]: registration_name(item)
        for item in bindings["registrations"]
        if registration_name(item) in columns
    }
    ownership = {
        name: item
        for item in bindings.get("parameter_ownership", ())
        for name in item["names"]
    }
    seen, reads, resources = set(), {}, set()
    for bound in path.tasks:
        positions = dict(
            zip(
                bound.description.get("tensor_positions", ()),
                bound.description.get("tensor_geometry", ()),
                strict=True,
            )
        )
        task_reads = []
        for slot in bound.entry.input_slots:
            is_gradient = (
                bound.task.phase == "optimizer" and slot.object_id in gradients
            )
            name = (
                gradients[slot.object_id]
                if is_gradient
                else weights.get(slot.object_id)
                if bound.task.phase in {"forward", "backward"}
                else None
            )
            if name is None:
                continue
            column = columns[name] + int(is_gradient)
            if column in seen:
                continue
            geometry = positions[slot.leaf_index]
            shape, stride = tuple(geometry["shape"]), tuple(geometry["stride"])
            elements, contribute = prod(shape), True
            if name in ownership:
                record = ownership[name]
                copies = GroupRef(name, record["copy_groups"]).members(logical_rank)
                if is_gradient:
                    stored = record["optimizer"]
                    owned = StorageOwnership(
                        stored["elements"],
                        tuple(
                            group for group in stored["groups"] if group[0] in copies
                        ),
                    )
                    if elements != owned.capacity:
                        raise ValueError(
                            f"norms require the declared update gradient: {name}"
                        )
                    _, elements = owned.region(logical_rank)
                    contribute = logical_rank == min(
                        owned.matching_owners(logical_rank)
                    )
                else:
                    if elements != prod(record["local_shape"]):
                        raise ValueError(
                            f"norms require the complete compute weight: {name}"
                        )
                    contribute = logical_rank == min(copies)
            dtype = geometry["dtype"]
            if dtype not in {
                "torch.float16",
                "torch.bfloat16",
                "torch.float32",
                "torch.float64",
            }:
                raise ValueError(f"parameter norms do not support {dtype}: {name}")
            task_reads.append(
                NormInput(
                    column, slot.leaf_index, elements, shape, stride, dtype, contribute
                )
            )
            seen.add(column)
        if task_reads:
            reads[bound.task.task_id] = tuple(task_reads)
            resources.add(bound.task.resource)
    missing = set(range(2 * len(names))) - seen
    if missing:
        labels = [
            f"{names[c // 2]}:{'gradient' if c % 2 else 'weight'}"
            for c in sorted(missing)
        ]
        raise ValueError(f"Plan has no safe norm observation point for {labels}")
    if len(resources) > 1:
        raise ValueError("parameter norms currently require one ordered task stream")
    return names, reads
