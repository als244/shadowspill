"""Owned, immutable ABI records with on-demand Python projections.

Only the copied bytes survive the call that captures them. No runtime pointer,
device marker, mutable native buffer or decoded-record cache is retained.
"""

from __future__ import annotations

import ctypes
import operator
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any, overload

import numpy as np


@cache
def _layout(record_type: type[ctypes.Structure]) -> np.dtype[Any]:
    fields = [(field[0], field[1]) for field in record_type._fields_]
    return np.dtype(
        {
            "names": [name for name, _ in fields],
            "formats": [f"u{ctypes.sizeof(kind)}" for _, kind in fields],
            "offsets": [getattr(record_type, name).offset for name, _ in fields],
            "itemsize": ctypes.sizeof(record_type),
        }
    )


@dataclass(frozen=True, slots=True, eq=False)
class PackedEventRecords[T](Sequence[T]):
    """An immutable event sequence; indexing/iteration decodes only on demand.

    Construction is internal: the caller validates sequence and enum columns
    before publishing these records. The ABI currently contains unsigned integer
    fields only. Layout comes from ctypes, including native padding and offsets.
    """

    _payload: bytes
    _record_type: type[ctypes.Structure]
    _decode: Callable[[Any], T]

    def __post_init__(self) -> None:
        if not isinstance(self._payload, bytes):
            raise TypeError("packed events require owned immutable bytes")
        if len(self._payload) % ctypes.sizeof(self._record_type):
            raise ValueError("packed event buffer contains a partial record")

    def __len__(self) -> int:
        return len(self._payload) // ctypes.sizeof(self._record_type)

    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[T, ...]: ...

    def __getitem__(self, index: int | slice) -> T | tuple[T, ...]:
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("event index out of range")
        record = self._record_type.from_buffer_copy(
            self._payload, index * ctypes.sizeof(self._record_type)
        )
        return self._decode(record)

    def __iter__(self) -> Iterator[T]:
        # Explicit full iteration may decode every record. Keep only this
        # temporary ABI copy, never a cached tuple of Python objects.
        records = (self._record_type * len(self)).from_buffer_copy(self._payload)
        return (self._decode(record) for record in records)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Sequence):
            return NotImplemented
        return len(self) == len(other) and all(
            left == right for left, right in zip(self, other, strict=True)
        )

    def __repr__(self) -> str:
        return f"PackedEventRecords(count={len(self)}, bytes={len(self._payload)})"

    def column(self, name: str) -> np.ndarray[Any, Any]:
        """A read-only raw field, with its native sentinel and integer dtype."""

        records = np.frombuffer(self._payload, dtype=_layout(self._record_type))
        return records[name]

    def matching(self, name: str, value: int) -> Iterator[T]:
        """Decode only matching records, preserving original sequence numbers."""

        indices = np.flatnonzero(self.column(name) == value)
        return (self[int(index)] for index in indices)
