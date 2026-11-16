"""Named persistent tensors and bounded initialization/export, outside execution."""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterator, Mapping
from typing import TYPE_CHECKING, Any, NamedTuple

import torch

from shadowspill.artifacts.examples import input_groups, single_forward
from shadowspill.artifacts.tensors import TensorChunk as TensorChunk
from shadowspill.artifacts.tensors import registration_name

from .initialization import fill_default
from .storage import dtype_of
from .tensor_layout import TensorLayout

if TYPE_CHECKING:
    from .storage import Storage


def tensor_entries(
    state: TensorLayout,
) -> Iterator[tuple[str, str, Mapping[str, Any] | None]]:
    bindings = state.plan.program.bindings
    for item in bindings["registrations"]:
        yield registration_name(item), item["object_id"], None
    optimizer_bindings = bindings.get("optimizer_bindings", ())
    mastered = {
        item["parameter_name"]
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
    if bindings["kind"] in {"forward_backward", "optim_step"}:
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
    for index, slots in enumerate(input_groups(bindings)):
        sources = (
            bindings["example_root_sources"][index]
            if "example_root_sources" in bindings
            else bindings["root_sources"]
        )
        for slot in slots:
            source = sources[slot["leaf_index"]]
            if source["kind"] == "CONSTANT_TENSOR":
                # Lifted names are local to each export. Different examples can
                # use the same name for different values or tensor geometries.
                scope = "" if single_forward(bindings) else f"example_{index}."
                yield (
                    "constant." + scope + source["name"],
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
    for index, slots in enumerate(input_groups(bindings)):
        sources = (
            bindings["example_root_sources"][index]
            if "example_root_sources" in bindings
            else bindings["root_sources"]
        )
        covered.update(
            state.aliases[s["object_id"]]
            for s in slots
            if sources[s["leaf_index"]]["kind"] == "USER_INPUT"
        )
    missing = state.initial - covered
    if missing:
        raise ValueError(
            f"Plan has initial tensors without a loader binding: {sorted(missing)}"
        )


class OwnedTensor(NamedTuple):
    """One persistent tensor of a Storage and the region this rank owns of it."""

    name: str
    object_id: str
    initializer: Mapping[str, Any] | None
    alias: str
    geometry: Mapping[str, Any]
    dtype: torch.dtype
    capacity: int
    region: Any | None
    offset: int
    length: int
    canonical: bool = False

    def chunk(self, start: int, prefix: str = "") -> TensorChunk:
        """The request for the owned region from ``start``, named with ``prefix``."""
        return TensorChunk(
            self.name if self.canonical else prefix + self.name,
            tuple(self.geometry["shape"]),
            self.dtype,
            self.offset + start,
            self.capacity if self.region is None else self.region.storage_elements,
            tuple(self.geometry["stride"]),
            self.geometry.get("storage_offset", 0),
            self.offset,
            self.length,
        )


def owned_tensors(
    state: TensorLayout, *, skip_shared: bool = False
) -> Iterator[OwnedTensor]:
    """Each persistent tensor of ``state`` once, by its first name, with the
    region this rank owns; a shared alias is left out when asked."""
    visited = set()
    for name, object_id, initializer in tensor_entries(state):
        alias = state.aliases[object_id]
        if skip_shared and alias in state.shared_aliases:
            continue
        if alias in visited or not state.sizes[alias]:
            continue
        visited.add(alias)
        if object_id not in state.geometry:
            raise ValueError(f"missing tensor geometry for {name}")
        geometry = state.geometry[object_id]
        dtype = dtype_of(geometry)
        capacity = state.spill_sizes[alias] // dtype.itemsize
        region = state.regions.get(object_id)
        if region is not None and capacity != region.capacity:
            raise ValueError(f"spill lease does not match the declared shard: {name}")
        yield OwnedTensor(
            name,
            object_id,
            initializer,
            alias,
            geometry if region is None else region.geometry,
            dtype,
            capacity,
            region,
            0 if region is None else region.offset,
            capacity if region is None else region.length,
            object_id in state.persistent_objects,
        )


def tensor_descriptors(
    state: TensorLayout, *, prefix: str = "", skip_shared: bool = False
) -> Iterator[TensorChunk]:
    """One chunk spanning each persistent tensor's owned region, named as
    ``visit_tensors`` names it: how a checkpoint describes a tensor, read
    without touching any data."""
    for tensor in owned_tensors(state, skip_shared=skip_shared):
        yield tensor.chunk(0, prefix)


def visit_tensors(
    state: Storage,
    callback: Callable[[TensorChunk, torch.Tensor], None],
    chunk_bytes: int,
    *,
    initialize: bool,
    restore: bool | Collection[str] = False,
    prefix: str = "",
    skip_shared: bool = False,
    names: Collection[str] | None = None,
) -> None:
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    if initialize and names is not None:
        raise ValueError("tensor selection applies only to export, not initialization")
    named = {name: object_id for name, object_id, _ in tensor_entries(state)}
    for tensor in owned_tensors(state, skip_shared=skip_shared):
        if names is not None and tensor.chunk(0, prefix).name not in names:
            continue
        if initialize and tensor.alias not in state.initial:
            continue
        if initialize and tensor.length < tensor.capacity:
            for _, padding in state.spill.chunks(
                tensor.alias,
                dtype=tensor.dtype,
                start=tensor.length,
                chunk_bytes=chunk_bytes,
                read=False,
                write=True,
            ):
                padding.zero_()
        for start, destination in state.spill.chunks(
            tensor.alias,
            dtype=tensor.dtype,
            length=tensor.length,
            chunk_bytes=chunk_bytes,
            read=not initialize,
            write=initialize,
        ):
            request = tensor.chunk(start, prefix)
            restoring = (
                restore if isinstance(restore, bool) else request.name in restore
            )
            if initialize and not restoring and tensor.name.startswith("gradient."):
                destination.zero_()
            elif (
                initialize
                and not restoring
                and tensor.initializer
                and tensor.initializer["kind"] != "external"
            ):
                fill_default(
                    state,
                    tensor.object_id,
                    destination,
                    start,
                    tensor.initializer,
                    named,
                )
            else:
                callback(request, destination)
