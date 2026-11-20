"""Portable tensor names and layouts shared by admission and execution."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .program import Program

if TYPE_CHECKING:
    import torch


def public_tensor_geometry(program: Program) -> dict[str, Mapping[str, Any]]:
    """Resolve public tensor geometry from the original saved compiler contracts.

    Output slots bind semantic leaves to Program object IDs. Input geometry and
    storage byte extents cannot substitute for the output's logical view shape.
    """
    from .examples import output_groups

    wanted = {name for group in output_groups(program.bindings) for name in group}
    wanted.discard(None)
    tasks = {task.task_id: task for task in program.tasks}
    result: dict[str, Mapping[str, Any]] = {}
    for entry in program.bindings["entrypoints"]:
        selected = [
            slot for slot in entry["output_slots"] if slot["object_id"] in wanted
        ]
        if not selected:
            continue
        task = tasks[entry["task_id"]]
        views = {
            view["leaf_index"]: view
            for view in program.bindings["manifests"][task.profile_id][
                "storage_contract"
            ]["output_views"]
        }
        for slot in selected:
            view = views.get(slot["leaf_index"])
            if view is None or view["layout"] != "torch.strided":
                raise ValueError("public tensor has no saved strided output geometry")
            geometry = {"shape": tuple(view["shape"]), "dtype": view["dtype"]}
            name = slot["object_id"]
            if result.setdefault(name, geometry) != geometry:
                raise ValueError("public tensor writers disagree on output geometry")
    if result.keys() != wanted:
        raise ValueError("public tensor has no saved producer geometry")
    return result


@dataclass(frozen=True)
class TensorChunk:
    """A contiguous region of one persistent tensor's storage.

    `Executor.initialize`, `Executor.export_tensors` and the checkpoint
    functions call the caller's callback with a chunk and a one-dimensional
    buffer of its elements; `Executor.tensor_chunks` describes each tensor
    whole. Offsets count dtype elements of the complete tensor's storage, not
    of a flattened view, so tied and strided views share one region. A worker
    of a sharded tensor sees only its own region; padding is never shown.

    Attributes
    ----------
    name : str
        The tensor's canonical name, such as a parameter's name,
        ``master.<parameter>``, ``gradient.<parameter>`` or an optimizer
        state's name. Tensors private to an additional Plan are named
        ``plans.<plan>.<tensor>``.
    shape : tuple of int
        Shape of the complete tensor.
    dtype : torch.dtype
        Element type.
    offset : int
        Storage offset of the chunk's first element.
    storage_elements : int
        Elements in the complete tensor's storage.
    stride : tuple of int
        The tensor's strides, in elements.
    storage_offset : int
        Storage offset of the tensor's first element.
    region_offset : int
        Storage offset where this worker's region begins; zero for a tensor
        the worker holds whole.
    region_elements : int or None
        Elements in this worker's region; None means the whole storage.
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


def export_selection(
    chunks: Iterable[TensorChunk], names: Collection[str] | None
) -> frozenset[str] | None:
    """Validate exact canonical export names before any tensor data is read."""
    if names is None:
        return None
    if isinstance(names, str | bytes) or any(
        not isinstance(name, str) for name in names
    ):
        raise TypeError("export names require a collection of tensor names")
    selected = frozenset(names)
    missing = selected - {chunk.name for chunk in chunks}
    if missing:
        raise ValueError(f"unknown export tensors: {sorted(missing)[:5]}")
    return selected


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


def model_storage_bytes(program: Program) -> dict[str, int]:
    """Registered storage extents, excluding padding added by distributed lowering."""
    objects = {item.object_id: item for item in program.objects}
    sizes: dict[str, int] = {}
    for entry in program.bindings.get("registrations", ()):
        item = objects[entry["object_id"]]
        sizes[item.alias_group_id] = max(
            sizes.get(item.alias_group_id, 0), item.offset_bytes + item.size_bytes
        )
    return sizes


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
