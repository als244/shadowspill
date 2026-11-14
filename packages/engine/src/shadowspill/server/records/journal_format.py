"""Bounded, canonical, hash-linked journal commits; never Python serialization."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from shadowspill.artifacts._values import encoded, freeze
from shadowspill.artifacts.archive import _json_object
from shadowspill_client.protocol.requests import identifier

_MAGIC = b"SSJ1\n"
_HEADER = len(_MAGIC) + 65


@dataclass(frozen=True)
class JournalLimits:
    record_bytes: int = 1 << 20
    commit_bytes: int = 8 << 20
    commit_records: int = 128

    def __post_init__(self) -> None:
        if (
            any(
                type(n) is not int or n < 1
                for n in (self.record_bytes, self.commit_bytes, self.commit_records)
            )
            or self.record_bytes > self.commit_bytes
        ):
            raise ValueError(
                "journal limits must be positive bounded record/commit sizes"
            )


@dataclass(frozen=True)
class JournalRecord:
    position: int
    kind: str
    data: Mapping[str, Any]

    def __post_init__(self) -> None:
        if type(self.position) is not int or self.position < 1:
            raise ValueError("journal positions start at one")
        identifier(self.kind, "journal kind")
        if not isinstance(self.data, Mapping):
            raise TypeError("journal data must be a metadata mapping")
        object.__setattr__(self, "data", freeze(self.data))


def encode_commit(
    index: int,
    position: int,
    previous: str | None,
    entries: Sequence[tuple[str, Mapping[str, Any]]],
    limits: JournalLimits,
) -> tuple[bytes, tuple[JournalRecord, ...], str]:
    if not 1 <= len(entries) <= limits.commit_records:
        raise ValueError("journal commit exceeds record count limit or is empty")
    records, rows, size = [], [], _HEADER
    for offset, (kind, value) in enumerate(entries):
        record = JournalRecord(position + offset, kind, value)
        row = {"kind": record.kind, "data": record.data}
        length = len(encoded(row))
        if length > limits.record_bytes:
            raise ValueError("journal record exceeds byte limit")
        size += length
        if size > limits.commit_bytes:
            raise ValueError("journal commit exceeds byte limit")
        rows.append(row)
        records.append(record)
    payload = encoded(
        {
            "format": "shadowspill.request-journal",
            "version": 1,
            "index": index,
            "position": position,
            "previous": previous,
            "entries": rows,
        }
    )
    checksum = sha256(payload).hexdigest()
    result = _MAGIC + checksum.encode("ascii") + b"\n" + payload
    if len(result) > limits.commit_bytes:
        raise ValueError("journal commit exceeds byte limit")
    return result, tuple(records), checksum


def decode_commit(
    data: bytes,
    *,
    index: int,
    position: int,
    previous: str | None,
    limits: JournalLimits,
) -> tuple[tuple[JournalRecord, ...], str]:
    if len(data) > limits.commit_bytes:
        raise ValueError("journal commit exceeds byte limit")
    if (
        len(data) <= _HEADER
        or not data.startswith(_MAGIC)
        or data[_HEADER - 1 : _HEADER] != b"\n"
    ):
        raise ValueError("invalid journal commit header")
    checksum = sha256(data[_HEADER:]).hexdigest()
    if data[len(_MAGIC) : _HEADER - 1] != checksum.encode("ascii"):
        raise ValueError("journal commit checksum mismatch")
    payload = _json_object(data[_HEADER:])
    if (
        set(payload)
        != {"format", "version", "index", "position", "previous", "entries"}
        or payload["format"] != "shadowspill.request-journal"
        or type(payload["version"]) is not int
        or payload["version"] != 1
        or type(payload["index"]) is not int
        or payload["index"] != index
        or type(payload["position"]) is not int
        or payload["position"] != position
        or payload["previous"] != previous
        or not isinstance(payload["entries"], list)
    ):
        raise ValueError("journal commit version, position or hash chain differs")
    if not 1 <= len(payload["entries"]) <= limits.commit_records:
        raise ValueError("journal commit exceeds record count limit or is empty")
    entries = []
    for row in payload["entries"]:
        if not isinstance(row, dict) or set(row) != {"kind", "data"}:
            raise ValueError("invalid journal record")
        entries.append((row["kind"], row["data"]))
    canonical, records, _ = encode_commit(index, position, previous, entries, limits)
    if canonical != data:
        raise ValueError("journal commit is not canonical")
    return records, checksum
