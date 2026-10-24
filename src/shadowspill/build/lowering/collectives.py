"""Declare framework collectives without compiling them or assigning timings."""

from typing import Any

from shadowspill.artifacts._values import digest
from shadowspill.artifacts.program import TaskDefinition
from shadowspill.communication.requests import CollectiveRequest

from .catalog import CompiledTasks


def declare_collective(
    description: dict[str, Any],
    compiled: CompiledTasks,
    descriptions: dict[str, Any],
) -> str:
    key = "collective_" + digest(description)
    compiled.definitions[key] = TaskDefinition(key, key, None, description["operation"])
    request = CollectiveRequest(
        description["operation"],
        tuple(description["groups"][0]),
        description["tensor_bytes"],
        description["dtype"],
        inplace=description.get("inplace"),
    )
    gathered = request.operation == "all_gather"
    shapes = [request.elements // len(request.ranks) if gathered else request.elements]
    if gathered and request.inplace:
        shapes.append(request.elements)
    compiled.inputs[key] = {
        "count": len(shapes),
        "tensor_positions": list(range(len(shapes))),
        "static_values": [],
        "tensor_geometry": [
            {
                "dtype": request.dtype,
                "shape": [size],
                "stride": [1],
                "storage_offset": 0,
            }
            for size in shapes
        ],
    }
    descriptions[key] = description
    return key
