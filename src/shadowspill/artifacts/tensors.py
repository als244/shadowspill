"""Portable tensor names and layouts shared by admission and execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .program import Program

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class TensorChunk:
    """A contiguous region of physical tensor storage, in dtype elements.

    ``shape`` describes the original tensor. ``offset`` indexes its storage,
    not a flattened logical view; tied/strided views share one initialization.
    Distributed loaders receive only this worker's valid region. ``offset`` is
    still relative to the complete tensor storage; physical padding is internal.
    """

    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    offset: int
    storage_elements: int
    stride: tuple[int, ...] = ()
    storage_offset: int = 0
    region_offset: int = 0
    region_elements: int | None = None


def tensor_geometry(program: Program) -> dict[str, Mapping[str, Any]]:
    result = dict(program.bindings.get("model_tensor_geometry", {}))
    result.update(program.bindings.get("gradient_geometry", {}))
    tasks = {task.task_id: task for task in program.tasks}
    for entry in program.bindings["entrypoints"]:
        description = program.bindings["task_inputs"][
            tasks[entry["task_id"]].profile_id
        ]
        geometry = dict(
            zip(
                description["tensor_positions"],
                description["tensor_geometry"],
                strict=True,
            )
        )
        for slot in entry["input_slots"]:
            result.setdefault(slot["object_id"], geometry[slot["leaf_index"]])
    for item in program.bindings.get("optimizer_bindings", ()):
        if item["object_id"] is not None:
            result.setdefault(item["object_id"], {**item, "storage_offset": 0})
    for item in program.bindings.get("persistent_gradients", ()):
        result.setdefault(
            item["gradient_object_id"],
            {
                "shape": item["shape"],
                "stride": (1,),
                "storage_offset": 0,
                "dtype": item["dtype"],
            },
        )
    return result


def registration_name(item: Mapping[str, Any]) -> str:
    parts = (item["name"], *item.get("component_path", ()))
    return ".".join(parts)


def shared_model_aliases(primary: Program, other: Program) -> dict[str, str]:
    """Match named model storage, including ties/views, without importing Torch."""
    registered = {
        registration_name(item): item for item in primary.bindings["registrations"]
    }
    geometry = tensor_geometry(primary), tensor_geometry(other)
    objects = ({o.object_id: o for o in p.objects} for p in (primary, other))
    source_objects, target_objects = objects
    source_aliases, target_aliases = (
        {a.alias_group_id: a for a in p.alias_groups} for p in (primary, other)
    )
    forward: dict[str, str] = {}
    reverse: dict[str, str] = {}
    for item in other.bindings["registrations"]:
        name = registration_name(item)
        source = registered.get(name)
        if source is None:
            raise ValueError(
                f"additional Plan model tensor is absent from default: {name}"
            )
        left, right = (
            source_objects[source["object_id"]],
            target_objects[item["object_id"]],
        )
        if source["parameter"] != item["parameter"]:
            raise ValueError(f"model tensor parameter/buffer role differs: {name}")
        views = geometry[0].get(left.object_id), geometry[1].get(right.object_id)
        if any(
            view is None or not {"dtype", "shape", "stride"}.issubset(view)
            for view in views
        ):
            raise ValueError(f"model tensor has no portable layout: {name}")
        a, b = views
        assert a is not None and b is not None
        if any(
            a.get(key, 0) != b.get(key, 0)
            for key in ("dtype", "shape", "stride", "storage_offset")
        ) or (left.offset_bytes, left.size_bytes) != (
            right.offset_bytes,
            right.size_bytes,
        ):
            raise ValueError(f"model tensor layout differs: {name}")
        src, dst = left.alias_group_id, right.alias_group_id
        sa, ta = source_aliases[src], target_aliases[dst]
        if (sa.size_bytes, sa.transfer) != (ta.size_bytes, ta.transfer):
            raise ValueError(f"model tensor storage differs: {name}")
        if forward.setdefault(dst, src) != src or reverse.setdefault(src, dst) != dst:
            raise ValueError(f"model tensor aliasing differs: {name}")
    return forward
