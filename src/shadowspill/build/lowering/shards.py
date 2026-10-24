"""Describe physical weight shards without inserting costs or live collectives."""

from dataclasses import replace
from typing import Any

from shadowspill.artifacts.program import Program
from shadowspill.parallelism.ownership import ParameterOwnership


def weight_shards(
    program: Program, parameters: tuple[ParameterOwnership, ...]
) -> Program:
    by_name = {name: parameter for parameter in parameters for name in parameter.names}
    aliases = {item.object_id: item.alias_group_id for item in program.objects}
    sizes = {item.alias_group_id: item.size_bytes for item in program.alias_groups}
    registered_sizes: dict[str, int] = {}
    object_by_id = {item.object_id: item for item in program.objects}
    for entry in program.bindings["registrations"]:
        item = object_by_id[entry["object_id"]]
        registered_sizes[item.alias_group_id] = max(
            registered_sizes.get(item.alias_group_id, 0),
            item.offset_bytes + item.size_bytes,
        )
    geometry = program.bindings["model_tensor_geometry"]
    shards: dict[str, Any] = {}
    for entry in program.bindings["registrations"]:
        parameter = by_name[entry["name"]]
        groups = parameter.weights.groups
        count = len(groups[0])
        object_id = entry["object_id"]
        alias = aliases[object_id]
        width = geometry[object_id]["element_size"]
        chunk = (sizes[alias] + width * count - 1) // (width * count) * width
        record = {
            "groups": groups,
            "shard_bytes": chunk,
            "logical_bytes": registered_sizes[alias],
            "padded_bytes": chunk * count,
            # The native weight gather operates on bytes, independently of
            # parameter representation. Padding still respects tensor elements.
            "dtype": "torch.uint8",
        }
        if alias in shards and shards[alias] != record:
            raise ValueError(
                f"tensors sharing {alias} have incompatible storage sharding"
            )
        shards[alias] = record
    shards = {
        alias: record
        for alias, record in shards.items()
        if len(record["groups"][0]) > 1
    }
    return replace(
        program,
        alias_groups=tuple(
            replace(alias, size_bytes=shards[alias.alias_group_id]["padded_bytes"])
            if alias.alias_group_id in shards
            else alias
            for alias in program.alias_groups
        ),
        bindings={**program.bindings, "weight_shards": shards},
    )
