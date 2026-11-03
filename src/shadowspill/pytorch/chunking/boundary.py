"""Materialize exact public storage while keeping execution scratch local."""

from __future__ import annotations

from typing import Any

import torch


def unique_writes(
    destination: torch.Tensor, source: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Copy a broadcast boundary once per physical element, keeping its aliases."""
    for dim, (size, stride) in enumerate(
        zip(destination.shape, destination.stride(), strict=True)
    ):
        if size > 1 and stride == 0:
            destination = destination.narrow(dim, 0, 1)
            source = source.narrow(dim, 0, 1)
    return destination, source


def boundary_values(
    metadata: dict[str, Any],
    arguments: tuple[torch.Tensor, ...],
    values: list[Any] | None = None,
    *,
    value_leaves: set[int] | None = None,
) -> list[Any]:
    """Restore captured aliases/offsets after independently compiled children.

    A child compiler can change strides, offsets, and physical allocation sizes.
    Reuse a fresh root only when its complete physical contract matches. The
    parent manifest must never undercharge a larger child-owned allocation.
    """
    views = metadata["storage"]["output_views"]
    specs = metadata["storage"]["roots"]
    devices = [x.device for x in arguments if isinstance(x, torch.Tensor)]
    if not devices and values is not None:
        devices = [x.device for x in values if isinstance(x, torch.Tensor)]
    if not devices:
        return list(values) if values is not None else [None] * metadata["output_count"]
    device = next((x for x in devices if x.type != "cpu"), devices[0])
    result: list[Any] = [None] * metadata["output_count"]
    for index, value in metadata.get("output_literals", ()):
        result[index] = value
    owned_storages = {x.untyped_storage()._cdata for x in arguments}
    for index, root in enumerate(specs):
        members = [view for view in views if view["root_id"] == index]
        if (
            values is not None
            and root["kind"] == "fresh"
            and members
            and (
                value_leaves is None
                or all(view["leaf_index"] in value_leaves for view in members)
            )
        ):
            tensors = [values[view["leaf_index"]] for view in members]
            storage = tensors[0].untyped_storage()
            if (
                storage._cdata not in owned_storages
                and storage.nbytes() == root["minimum_span_bytes"]
                and all(
                    tensor.untyped_storage()._cdata == storage._cdata
                    and tuple(tensor.shape) == tuple(view["shape"])
                    and tuple(tensor.stride()) == tuple(view["stride"])
                    and tensor.storage_offset() * tensor.element_size()
                    == view["offset_bytes"]
                    and str(tensor.dtype) == view["dtype"]
                    for tensor, view in zip(tensors, members, strict=True)
                )
            ):
                for tensor, view in zip(tensors, members, strict=True):
                    result[view["leaf_index"]] = tensor
                owned_storages.add(storage._cdata)
                continue
        owner = (
            arguments[root["source_input"]]
            if root["kind"] == "input"
            else torch.empty(
                root["minimum_span_bytes"],
                dtype=torch.uint8,
                device="cpu"
                if index in metadata.get("host_output_roots", ())
                else device,
            )
        )
        for view in members:
            dtype = getattr(torch, view["dtype"].removeprefix("torch."))
            element = torch.empty((), dtype=dtype, device="meta").element_size()
            offset = view["offset_bytes"] // element
            if root["kind"] == "input":
                position = root["source_input"]
                offset += owner.storage_offset() - metadata["input_offsets"][position]
            typed = owner if owner.dtype == dtype else owner.view(dtype)
            destination = typed.as_strided(view["shape"], view["stride"], offset)
            result[view["leaf_index"]] = destination
            if (
                values is not None
                and root["kind"] == "fresh"
                and (value_leaves is None or view["leaf_index"] in value_leaves)
            ):
                target, source = unique_writes(destination, values[view["leaf_index"]])
                target.copy_(source)
    return result
