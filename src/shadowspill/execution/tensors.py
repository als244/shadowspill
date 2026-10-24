"""Named persistent tensors and bounded initialization/export, outside execution."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from shadowspill.artifacts.tensors import registration_name

from .initialization import fill_default
from .storage import dtype_of

if TYPE_CHECKING:
    from .storage import Storage


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


def tensor_entries(
    state: Storage,
) -> Iterator[tuple[str, str, Mapping[str, Any] | None]]:
    bindings = state.plan.program.bindings
    for item in bindings["registrations"]:
        yield registration_name(item), item["object_id"], None
    optimizer_bindings = bindings.get("optimizer_bindings", ())
    mastered = {
        item["name"].removeprefix("compute.")
        for item in optimizer_bindings
        if item["role"] == "compute_copy"
    }
    for item in optimizer_bindings:
        if item["role"] == "parameter" and item["name"] in mastered:
            yield (
                "master." + item["name"],
                item["object_id"],
                item.get("initialization"),
            )
        if (
            item["role"] not in {"parameter", "gradient"}
            and item["object_id"] is not None
        ):
            yield item["name"], item["object_id"], item.get("initialization")
    if bindings["kind"] == "forward_backward":
        for item in bindings.get("persistent_gradients", bindings["gradients"]):
            yield "gradient." + item["parameter_name"], item["gradient_object_id"], None
    for item in bindings.get("fixed_inputs", ()):
        yield (
            "fixed." + item["object_id"],
            item["object_id"],
            {
                "kind": "constant",
                "value": item["fill"],
            },
        )
    groups = bindings["root_inputs"]
    if bindings["kind"] == "forward":
        groups = [groups]
    for slots in groups:
        for slot in slots:
            source = bindings["root_sources"][slot["leaf_index"]]
            if source["kind"] == "CONSTANT_TENSOR":
                yield (
                    "constant." + source["name"],
                    slot["object_id"],
                    (
                        {"kind": "value", "value": source["value"]}
                        if "value" in source
                        else None
                    ),
                )


def verify_initialization(state: Storage) -> None:
    covered = {state.aliases[object_id] for _, object_id, _ in tensor_entries(state)}
    bindings = state.plan.program.bindings
    groups = bindings["root_inputs"]
    if bindings["kind"] == "forward":
        groups = [groups]
    for slots in groups:
        covered.update(
            state.aliases[s["object_id"]]
            for s in slots
            if bindings["root_sources"][s["leaf_index"]]["kind"] == "USER_INPUT"
        )
    missing = state.initial - covered
    if missing:
        raise ValueError(
            f"Plan has initial tensors without a loader binding: {sorted(missing)}"
        )


def visit_tensors(
    state: Storage,
    callback: Callable[[TensorChunk, torch.Tensor], None],
    chunk_bytes: int,
    *,
    initialize: bool,
    restore: bool = False,
    prefix: str = "",
    skip_shared: bool = False,
) -> None:
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    visited = set()
    entries = tuple(tensor_entries(state))
    named = {name: object_id for name, object_id, _ in entries}
    for name, object_id, initializer in entries:
        alias = state.aliases[object_id]
        if skip_shared and alias in state.shared_aliases:
            continue
        if initialize and alias not in state.initial:
            continue
        if alias in visited or not state.sizes[alias]:
            continue
        visited.add(alias)
        if object_id not in state.geometry:
            raise ValueError(f"missing tensor geometry for {name}")
        geometry = state.geometry[object_id]
        dtype = dtype_of(geometry)
        owner = state.host(alias).view(dtype)
        region = state.regions.get(object_id)
        if region is not None and owner.numel() != region.capacity:
            raise ValueError(f"spill lease does not match the declared shard: {name}")
        length = owner.numel() if region is None else region.length
        offset = 0 if region is None else region.offset
        if region is not None:
            geometry = region.geometry
        if initialize:
            owner[length:].zero_()
        width = max(1, chunk_bytes // dtype.itemsize)
        for start in range(0, length, width):
            destination = owner[start : min(start + width, length)]
            request = TensorChunk(
                prefix + name,
                tuple(geometry["shape"]),
                dtype,
                offset + start,
                owner.numel() if region is None else region.storage_elements,
                tuple(geometry["stride"]),
                geometry.get("storage_offset", 0),
            )
            if initialize and not restore and name.startswith("gradient."):
                destination.zero_()
            elif (
                initialize
                and not restore
                and initializer
                and initializer["kind"] != "external"
            ):
                fill_default(state, object_id, destination, start, initializer, named)
            else:
                callback(request, destination)
