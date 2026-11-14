"""Portable execution of locally live regions within a checkpoint task."""

from __future__ import annotations

import pickle
from typing import Any

import torch
from torch.utils._pytree import tree_flatten

from .boundary import boundary_values

MAGIC = b"SHADOWSPILL-REGION-COMPOSITION-1\n"


class RegionExecutable:
    def __init__(self, metadata: dict[str, Any], children: dict[int, Any]) -> None:
        self.metadata = metadata
        self.children = children

    def serialize(self) -> bytes:
        return MAGIC + pickle.dumps(
            {
                "metadata": self.metadata,
                "children": {
                    index: child.serialize() for index, child in self.children.items()
                },
            },
            protocol=5,
        )

    @classmethod
    def deserialize(cls, payload: bytes) -> RegionExecutable:
        from shadowspill.pytorch.executable import restore_executable

        stored = pickle.loads(payload[len(MAGIC) :])
        return cls(
            stored["metadata"],
            {
                index: restore_executable(code)
                for index, code in stored["children"].items()
            },
        )

    @torch.no_grad()
    def __call__(self, *arguments: torch.Tensor) -> tuple[Any, ...]:
        slots: list[Any] = [None] * self.metadata["slots"]
        slots[: len(arguments)] = arguments
        for index, instruction in enumerate(self.metadata["regions"]):
            inputs = tuple(slots[position] for position in instruction["inputs"])
            values, _ = tree_flatten(self.children[index](*inputs))
            outputs = boundary_values(instruction["boundary"], inputs, values)
            for position, value in zip(instruction["outputs"], outputs, strict=True):
                slots[position] = value
            # Clearing the slots alone is insufficient: these local containers
            # otherwise retain a complete region through the next child call.
            if outputs:
                del value
            del values, outputs, inputs
            for position in instruction["release"]:
                slots[position] = None
        values = [
            slots[item["slot"]] if "slot" in item else item["literal"]
            for item in self.metadata["outputs"]
        ]
        return tuple(boundary_values(self.metadata["boundary"], arguments, values))
