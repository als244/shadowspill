"""Portable sequential composition of finalized explicit chunk entrypoints."""

from __future__ import annotations

import pickle
from typing import Any

import torch
from torch.utils._pytree import tree_flatten

from .boundary import boundary_values, unique_writes
from .geometry import REDUCED, ChunkAxis

MAGIC = b"SHADOWSPILL-ROW-COMPOSITION-5\n"


class ChunkedExecutable:
    """Run a bounded chunk loop inside one ordinary task allocation scope."""

    def __init__(
        self,
        metadata: dict[str, Any],
        children: dict[int, Any],
        accumulating_children: dict[int, Any],
    ) -> None:
        self.metadata = metadata
        self.children = children
        self.accumulating_children = accumulating_children
        self.accumulation_leaves = metadata["accumulation_leaves"]
        self.inputs = tuple(
            ChunkAxis(**x) if isinstance(x, dict) else x for x in metadata["inputs"]
        )
        self.outputs = tuple(
            ChunkAxis(**x) if isinstance(x, dict) else x for x in metadata["outputs"]
        )
        self.mutated_leaves = {
            item["replacement_output_leaf"] for item in metadata["storage"]["mutations"]
        }

    def serialize(self) -> bytes:
        return MAGIC + pickle.dumps(
            {
                "metadata": self.metadata,
                "children": {
                    size: child.serialize() for size, child in self.children.items()
                },
                "accumulating_children": {
                    size: child.serialize()
                    for size, child in self.accumulating_children.items()
                },
            },
            protocol=5,
        )

    @classmethod
    def deserialize(cls, payload: bytes) -> ChunkedExecutable:
        from shadowspill.pytorch.executable import restore_executable

        stored = pickle.loads(payload[len(MAGIC) :])
        return cls(
            stored["metadata"],
            {
                size: restore_executable(code)
                for size, code in stored["children"].items()
            },
            {
                size: restore_executable(code)
                for size, code in stored["accumulating_children"].items()
            },
        )

    @torch.no_grad()
    def __call__(self, *arguments: torch.Tensor) -> tuple[Any, ...]:
        metadata = self.metadata
        rows, size = metadata["rows"], metadata["chunk_size"]
        result = None
        for start in range(0, rows, size):
            count = min(size, rows - start)
            # Keep dense permutation order: a captured transpose+view may rely
            # on it even when the logical chunk axis is not the second dimension.
            chunk_inputs = tuple(
                axis.slice(value, rows, start, count).clone(
                    memory_format=torch.preserve_format
                )
                if isinstance(axis, ChunkAxis)
                else torch.zeros_like(value)
                if axis == REDUCED and start != 0
                else value
                for axis, value in zip(self.inputs, arguments, strict=True)
            )
            if start and self.accumulation_leaves:
                assert result is not None
                values, _ = tree_flatten(
                    self.accumulating_children[count](
                        *chunk_inputs, *(result[i] for i in self.accumulation_leaves)
                    )
                )
            else:
                values, _ = tree_flatten(self.children[count](*chunk_inputs))
            if result is None:
                # A first gradient contribution already owns a full-sized
                # buffer. Adopt it if its exact public storage contract agrees;
                # allocate assembly storage only for sliced outputs.
                result = boundary_values(
                    metadata,
                    arguments,
                    values,
                    value_leaves={
                        i
                        for i, axis in enumerate(self.outputs)
                        if not isinstance(axis, ChunkAxis)
                    },
                )
            self._assemble(result, values, start, count)
            # Child results may alias its input copies. Drop both before the
            # next call so the allocator can reuse all chunk-local storage.
            del values, chunk_inputs
        assert result is not None
        return tuple(result)

    def _assemble(
        self, result: list[Any], values: list[Any], start: int, count: int
    ) -> None:
        storage = self.metadata["storage"]
        for view in storage["output_views"]:
            leaf = view["leaf_index"]
            destination, value, axis = result[leaf], values[leaf], self.outputs[leaf]
            if start and leaf in self.accumulation_leaves:
                continue
            if storage["roots"][view["root_id"]]["kind"] == "input" and (
                leaf not in self.mutated_leaves or not isinstance(axis, ChunkAxis)
            ):
                continue
            if isinstance(axis, ChunkAxis):
                target = axis.view(destination, self.metadata["rows"]).narrow(
                    axis.dim + 1, start, count
                )
                target, source = unique_writes(target, axis.view(value, count))
                target.copy_(source)
            elif start and axis == REDUCED:
                target, source = unique_writes(destination, value)
                target.add_(source)
