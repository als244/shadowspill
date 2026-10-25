"""Logical tensor ranges stored by a worker, separate from compute-buffer views."""

from collections.abc import Mapping
from dataclasses import dataclass
from math import prod
from typing import Any

import torch

from shadowspill.artifacts.program import Program
from shadowspill.artifacts.tensors import tensor_geometry
from shadowspill.parallelism import GroupRef


@dataclass(frozen=True)
class TensorRegion:
    geometry: Mapping[str, Any]
    storage_elements: int
    offset: int
    length: int
    capacity: int


def _region(
    geometry: Mapping[str, Any], elements: int, capacity: int, index: int
) -> TensorRegion:
    start = min(index * capacity, elements)
    return TensorRegion(
        geometry, elements, start, min(capacity, elements - start), capacity
    )


def tensor_regions(program: Program, logical_rank: int) -> dict[str, TensorRegion]:
    """Keep loader offsets in the original tensor, excluding physical padding.

    Compiled task geometry stays shard-local. These descriptions apply only to
    bounded host loading/export; no tensor data or process group is created.
    """
    if program.parallelism is None or program.parallelism.world_size == 1:
        return {}
    bindings = program.bindings
    geometry = tensor_geometry(program)
    objects = {item.object_id: item for item in program.objects}
    regions = {}
    for item in bindings.get("registrations", ()):
        value = item["object_id"]
        shard = bindings.get("weight_shards", {}).get(objects[value].alias_group_id)
        if shard is None:
            continue
        layout = geometry[value]
        width = getattr(torch, layout["dtype"].removeprefix("torch.")).itemsize
        if shard["shard_bytes"] % width or shard["logical_bytes"] % width:
            raise ValueError("weight shard must contain complete tensor elements")
        regions[value] = _region(
            layout,
            shard["logical_bytes"] // width,
            shard["shard_bytes"] // width,
            GroupRef("weights", shard["groups"]).index(logical_rank),
        )

    parameters = {
        name: item
        for item in bindings.get("parameter_ownership", ())
        for name in item["names"]
    }
    for item in bindings.get("optimizer_bindings", ()):
        value = item["object_id"]
        if value is None or value in regions or not item["shape"]:
            continue  # CPU controls and scalar state are not parameter shards.
        name = item.get("parameter_name")
        if name is not None and name in parameters:
            regions[value] = _parameter_region(
                parameters[name], "optimizer", item, logical_rank
            )
    if bindings.get("kind") == "forward_backward":
        for item in bindings.get("persistent_gradients", ()):
            regions[item["gradient_object_id"]] = _parameter_region(
                parameters[item["parameter_name"]], "gradients", item, logical_rank
            )
    return regions


def _parameter_region(
    parameter: Mapping[str, Any],
    component: str,
    binding: Mapping[str, Any],
    logical_rank: int,
) -> TensorRegion:
    owner = parameter[component]
    group = GroupRef(component, owner["groups"])
    elements = owner["elements"]
    capacity = (elements + group.size - 1) // group.size
    if group.size == 1:
        # Unsharded optimizer tensors retain their original tensor shape.
        if prod(binding["shape"]) != elements:
            raise ValueError("optimizer tensor is not parameter-element storage")
        shape = tuple(binding["shape"])
    else:
        expected = (elements,) if component == "gradients" else (capacity,)
        if tuple(binding["shape"]) != expected:
            raise ValueError(
                f"sharded {component} needs its declared parameter-element layout"
            )
        shape = tuple(parameter["local_shape"])
    stride = tuple(prod(shape[index + 1 :]) for index in range(len(shape)))
    return _region(
        {
            "shape": shape,
            "stride": stride,
            "storage_offset": 0,
            "dtype": binding["dtype"],
        },
        elements,
        capacity,
        group.index(logical_rank),
    )
