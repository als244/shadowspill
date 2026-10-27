"""Per-call scalar fills of named model buffers, through bounded spill access."""

from collections.abc import Mapping
from math import prod
from typing import Any

import torch

from .storage import Storage, dtype_of
from .tensor_spans import spans


class BufferControls:
    def __init__(self, storage: Storage):
        self.storage = storage
        self.bindings = {
            item["name"]: item
            for item in storage.plan.program.bindings.get("registrations", ())
            if not item["parameter"]
            and item.get("persistent", False)
            and not item.get("component_path")
        }

    def prepare(self, values: Mapping[str, Any]) -> list[tuple[str, Any]]:
        """Validate all fills before the caller writes any runtime controls."""
        prepared: list[tuple[str, Any]] = []
        seen: dict[str, torch.Tensor] = {}
        for name, binding in self.bindings.items():
            if name not in values:
                continue
            value = values[name]
            if not isinstance(value, int | float):
                raise TypeError(f"model buffer {name} takes one scalar number")
            object_id = binding["object_id"]
            geometry = self.storage.geometry[object_id]
            # Scalar conversion catches range errors without allocating a buffer copy.
            scalar = torch.full((), value, dtype=dtype_of(geometry), device="cpu")
            if object_id in seen and not torch.equal(scalar, seen[object_id]):
                raise ValueError(f"conflicting values for tied model buffer {name}")
            seen[object_id] = scalar
            self._geometry(object_id)
            prepared.append((object_id, scalar.item()))
        return prepared

    def _geometry(self, object_id: str) -> Mapping[str, Any]:
        geometry = self.storage.geometry[object_id]
        region = self.storage.regions.get(object_id)
        if region is not None:
            shape, stride = geometry["shape"], geometry["stride"]
            contiguous = all(
                size <= 1 or step == prod(shape[index + 1 :])
                for index, (size, step) in enumerate(zip(shape, stride, strict=True))
            )
            if (
                not contiguous
                or geometry.get("storage_offset", 0)
                or prod(shape) != region.storage_elements
            ):
                raise ValueError(
                    "sharded buffer controls require a dense full-storage view"
                )
            return {**geometry, "shape": (region.length,), "stride": (1,)}
        return geometry

    def apply(self, prepared: list[tuple[str, Any]]) -> None:
        for object_id, value in prepared:
            geometry = self._geometry(object_id)
            for _, start, count in spans(
                geometry["shape"], geometry["stride"], geometry.get("storage_offset", 0)
            ):
                for _, destination in self.storage.spill.chunks(
                    self.storage.aliases[object_id],
                    dtype=dtype_of(geometry),
                    start=start,
                    length=count,
                    read=False,
                    write=True,
                ):
                    destination.fill_(value)
