"""Validated tensor views; dtype conversion waits for admitted packed buffers."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from math import prod
from typing import Any

import numpy as np
from numpy.typing import NDArray

from shadowspill_client.protocol.inputs import TensorField
from shadowspill_client.protocol.tensors import TensorPayload, storage_dtype

CHUNK_ELEMENTS = 8192
# Fixed scratch allowance for numeric checks/conversion, independent of batch size.
VALIDATION_SCRATCH_BYTES = 512 << 10


def numeric_values(tensor: TensorPayload) -> Iterator[NDArray[Any]]:
    values = tensor.array().reshape(-1)
    for offset in range(0, len(values), CHUNK_ELEMENTS):
        block = values[offset : offset + CHUNK_ELEMENTS]
        yield (
            (block.astype(np.uint32) << 16).view(np.float32)
            if tensor.dtype == "bfloat16"
            else block
        )


def validate_values(tensor: TensorPayload, field: TensorField, label: str) -> None:
    target = storage_dtype(field.dtype)
    for block in numeric_values(tensor):
        if field.finite and not np.isfinite(block).all():
            raise ValueError(f"{label}: values must be finite")
        if field.minimum is not None and (block < field.minimum).any():
            raise ValueError(f"{label}: values below minimum {field.minimum}")
        if field.maximum is not None and (block > field.maximum).any():
            raise ValueError(f"{label}: values above maximum {field.maximum}")
        if field.dtype == "bool":
            if not ((block == 0) | (block == 1)).all():
                raise ValueError(f"{label}: boolean values must be 0 or 1")
        elif "int" in field.dtype:
            bounds = np.iinfo(target)
            if block.dtype.kind == "f":
                # An int64 upper bound rounds to 2**63 as float64; use the
                # exclusive power-of-two bound to reject that overflow exactly.
                bad = (
                    ~np.isfinite(block)
                    | (block != np.floor(block))
                    | (block < bounds.min)
                    | (block >= int(bounds.max) + 1)
                ).any()
            else:
                bad = int(block.min()) < bounds.min or int(block.max()) > bounds.max
            if bad:
                raise ValueError(
                    f"{label}: values cannot be represented as {field.dtype}"
                )
        elif field.finite:
            maximum = (
                float.fromhex("0x1.fep127")
                if field.dtype == "bfloat16"
                else np.finfo(target).max
            )
            if ((block > maximum) | (block < -maximum)).any():
                raise ValueError(f"{label}: values overflow {field.dtype}")


@dataclass(frozen=True)
class ValidatedTensor:
    dtype: str
    shape: tuple[int, ...]
    source: TensorPayload | None = None
    default: bool | int | float | None = None

    @property
    def elements(self) -> int:
        return prod(self.shape)

    def count_valid(self) -> int:
        if self.dtype != "bool":
            raise TypeError("participation requires a validated boolean field")
        if self.source is None:
            return self.elements if self.default else 0
        return sum(
            int(np.count_nonzero(block)) for block in numeric_values(self.source)
        )

    def copy_into(self, destination: NDArray[Any]) -> None:
        """Fill already admitted storage, including noncontiguous packing views.

        BF16 destinations use uint16 storage; conversion uses round-to-nearest,
        ties-to-even. No full intermediate cast or per-token Python objects.
        """
        if tuple(destination.shape) != self.shape or destination.dtype != storage_dtype(
            self.dtype
        ):
            raise ValueError("packed destination differs from validated shape/dtype")
        blocks: Iterable[NDArray[Any]]
        if self.source is None:
            blocks = (np.asarray([self.default]),)
        elif self.source.dtype == self.dtype:
            np.copyto(destination, self.source.array(), casting="no")
            return
        else:
            blocks = numeric_values(self.source)
        offset = 0
        for block in blocks:
            if self.dtype == "bfloat16":
                bits = block.astype(np.float32).view(np.uint32)
                converted = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(
                    np.uint16
                )
                converted = np.where(np.isnan(block), np.uint16(0x7FC0), converted)
            else:
                converted = block.astype(storage_dtype(self.dtype), copy=False)
            if self.source is None:
                destination.fill(converted[0])
            else:
                destination.flat[offset : offset + len(block)] = converted
                offset += len(block)
