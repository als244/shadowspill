"""Typed, shared JSON values in portable contiguous binary tables.

The 40-byte little-endian header names node/edge/string byte counts and a root.
Each 16-byte node stores kind, zero reserved bits, length and offset/value bits.
Array/object edges are uint32 references to earlier nodes; object edges alternate
string keys and values. Strings and large integers occupy the final byte table.
No executable Python records, class names, pickle or machine addresses are saved.
"""

from __future__ import annotations

import json
import math
import struct
from array import array
from collections.abc import Mapping
from typing import Any, cast

import numpy as np

MAGIC = b"SSMETA\x00\x01"
HEADER = struct.Struct("<8sQQQQ")
NODE = np.dtype(
    [
        ("kind", "u1"),
        ("flags", "u1"),
        ("reserved", "<u2"),
        ("length", "<u4"),
        ("offset", "<u8"),
    ]
)
NULL, FALSE, TRUE, INTEGER, FLOAT, STRING, ARRAY, OBJECT, BIGINT = range(9)
_U64 = (1 << 64) - 1
_U32 = (1 << 32) - 1


def encode(value: dict[str, Any]) -> bytes:
    """Visit shared containers once, then intern equal values by content.

    Numeric types and float bit patterns stay exact. Input references are retained
    until encoding completes; no persistent cache or Python addresses are saved.
    """
    rows: list[tuple[int, int, int, int, int]] = []
    edges = array("I")
    strings = bytearray()
    seen: dict[Any, int] = {}
    identities: dict[int, tuple[Any, int]] = {}
    active: set[int] = set()
    heights: list[int] = []

    def visit(item: Any, depth: int = 0) -> int:
        if depth > 128:
            raise ValueError("metadata nesting exceeds 128")
        kind = type(item)
        mapping = kind is dict or (
            kind not in (list, tuple, str, int, float, bool, type(None))
            and isinstance(item, Mapping)
        )
        container = mapping or kind in (list, tuple)
        identity = id(item)
        if container:
            cached = identities.get(identity)
            if cached is not None:
                index = cached[1]
                if depth + heights[index] > 128:
                    raise ValueError("metadata nesting exceeds 128")
                return index
            if identity in active:
                raise ValueError("metadata nesting is cyclic")
            active.add(identity)
        refs: tuple[int, ...] = ()
        key: tuple[int, Any]
        if mapping:
            if any(not isinstance(key, str) for key in item):
                raise TypeError("metadata object keys must be strings")
            refs = tuple(
                ref
                for key, child in sorted(item.items())
                for ref in (visit(key, depth + 1), visit(child, depth + 1))
            )
            key = OBJECT, refs
        elif kind in (list, tuple):
            refs = tuple(visit(child, depth + 1) for child in item)
            key = ARRAY, refs
        elif item is None:
            key = NULL, None
        elif kind is bool:
            key = TRUE if item else FALSE, None
        elif kind is int:
            key = INTEGER if -(1 << 63) <= item < 1 << 63 else BIGINT, item
        elif kind is float and math.isfinite(item):
            key = FLOAT, struct.unpack("<Q", struct.pack("<d", item))[0]
        elif kind is str:
            key = STRING, item
        elif isinstance(item, str):
            return visit(str.__str__(item), depth)
        elif isinstance(item, int):
            return visit(int.__int__(item), depth)
        elif isinstance(item, float) and kind is not float:
            return visit(float.__float__(item), depth)
        else:
            raise TypeError("metadata requires finite JSON values")
        found = seen.get(key)
        if found is not None:
            if container:
                identities[identity] = item, found
                active.remove(identity)
            return found
        tag = key[0]
        height = 1 + max(heights[ref] for ref in refs) if refs else 0
        if depth + height > 128:
            raise ValueError("metadata nesting exceeds 128")
        length = offset = 0
        if tag in (ARRAY, OBJECT):
            length, offset = len(refs), len(edges)
            edges.extend(refs)
        elif tag in (STRING, BIGINT):
            raw = str(item).encode("utf-8", errors="surrogatepass")
            length, offset = len(raw), len(strings)
            strings.extend(raw)
        elif tag in (INTEGER, FLOAT):
            offset = cast(int, key[1]) & _U64
        if length > _U32 or len(rows) >= _U32:
            raise ValueError("metadata table exceeds its index range")
        index = len(rows)
        rows.append((tag, 0, 0, length, offset))
        heights.append(height)
        seen[key] = index
        if container:
            identities[identity] = item, index
            active.remove(identity)
        return index

    root = visit(value)
    if edges.itemsize != 4:
        raise RuntimeError("metadata requires 32-bit unsigned array elements")
    edge_values = np.asarray(edges, dtype="<u4")
    return b"".join(
        (
            HEADER.pack(MAGIC, len(rows), len(edges), len(strings), root),
            np.asarray(rows, dtype=NODE).tobytes(),
            edge_values.tobytes(),
            strings,
        )
    )


def decode(
    data: bytes, *, expanded_limit: int | None
) -> tuple[dict[str, Any], list[Any]]:
    """Validate all table geometry before constructing each unique value once.

    Expanded size/depth limits reject small cyclic or exponentially expanding
    inputs before any artifact readers, hashes or inspection can traverse them.
    Array storage remains contiguous while decoding; repeated edges reuse the
    same owned input and subsequently the same immutable artifact record.
    """
    if len(data) < HEADER.size:
        raise ValueError("truncated shared metadata header")
    magic, count, edge_count, byte_count, root = HEADER.unpack_from(data)
    if magic != MAGIC or not 0 < count <= _U32 or root != count - 1:
        raise ValueError("unsupported shared metadata header")
    edge_start = HEADER.size + count * NODE.itemsize
    byte_start = edge_start + edge_count * 4
    if byte_start + byte_count != len(data):
        raise ValueError("shared metadata table sizes differ from the payload")
    nodes = np.frombuffer(data, dtype=NODE, count=count, offset=HEADER.size)
    edges = np.frombuffer(data, dtype="<u4", count=edge_count, offset=edge_start)
    if np.any(nodes["flags"]) or np.any(nodes["reserved"]):
        raise ValueError("unknown shared metadata flags")
    values: list[Any] = []
    sizes: list[int] = []
    depths: list[int] = []
    for index, row in enumerate(nodes):
        value: Any
        kind, length, offset = int(row["kind"]), int(row["length"]), int(row["offset"])
        depth = 0
        if kind in (NULL, FALSE, TRUE, INTEGER, FLOAT):
            if length or (kind in (NULL, FALSE, TRUE) and offset):
                raise ValueError("invalid shared metadata scalar")
            if kind == INTEGER:
                value = offset if offset < 1 << 63 else offset - (1 << 64)
            elif kind == FLOAT:
                value = struct.unpack("<d", struct.pack("<Q", offset))[0]
                if not math.isfinite(value):
                    raise ValueError("nonfinite shared metadata scalar")
            else:
                value = (None, False, True)[kind]
            size = len(json.dumps(value, allow_nan=False))
        elif kind in (STRING, BIGINT):
            if offset + length > byte_count:
                raise ValueError("shared metadata string lies outside its table")
            value = data[byte_start + offset : byte_start + offset + length].decode(
                "utf-8", errors="surrogatepass"
            )
            if kind == BIGINT:
                number = int(value)
                if str(number) != value or -(1 << 63) <= number < 1 << 63:
                    raise ValueError("invalid shared metadata large integer")
                value = number
            size = len(json.dumps(value, ensure_ascii=True))
        elif kind in (ARRAY, OBJECT):
            if offset + length > edge_count or (kind == OBJECT and length % 2):
                raise ValueError("shared metadata edges lie outside their table")
            refs = edges[offset : offset + length].tolist()
            if any(ref >= index for ref in refs):
                raise ValueError("shared metadata requires backward references")
            depth = 1 + max((depths[ref] for ref in refs), default=0)
            if kind == ARRAY:
                value = [values[ref] for ref in refs]
                size = 2 + max(0, length - 1) + sum(sizes[ref] for ref in refs)
            else:
                value = {}
                for key_ref, value_ref in zip(refs[::2], refs[1::2], strict=True):
                    key = values[key_ref]
                    if type(key) is not str or key in value:
                        raise ValueError(
                            "shared metadata has invalid or duplicate keys"
                        )
                    value[key] = values[value_ref]
                size = (
                    2
                    + max(0, length // 2 - 1)
                    + length // 2
                    + sum(sizes[ref] for ref in refs)
                )
        else:
            raise ValueError("unknown shared metadata node kind")
        if depth > 128 or (expanded_limit is not None and size > expanded_limit):
            raise ValueError("shared metadata exceeds expanded size/depth limit")
        values.append(value)
        sizes.append(size)
        depths.append(depth)
    result = values[root]
    if type(result) is not dict:
        raise ValueError("shared metadata root must be an object")
    return result, values
