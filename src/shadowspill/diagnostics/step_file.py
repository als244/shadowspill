"""A traced step on disk: its diagnostics as JSON, its event records as arrays.

The JSON file is `StepDiagnostics.as_dict(events=False)`: every view, without
the two event ledgers. Those grow with the step -- a step of a hundred
thousand tasks records millions of events -- so they are written beside it,
in `<name>.events.npz`, one column per field and no Python object per record
on the way out. A column holds its field as the runtime recorded it, the
runtime's sentinel for "none" included, and reading one back decodes it the
way the live trace is decoded.
"""

from __future__ import annotations

import json
from collections import namedtuple
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from shadowspill.diagnostics.step import StepDiagnostics
from shadowspill.runtime.telemetry import (
    NO_ID,
    CapturedAllocationEvent,
    decode_allocation_events,
)
from shadowspill.runtime.trace import (
    NO_STREAM_TIME,
    RuntimeTraceEvent,
    decode_runtime_events,
)
from shadowspill.schema import artifact_schema
from shadowspill.store import atomic_json, atomic_write

#: What follows a step file's stem in the name of the file holding its events.
EVENTS_SUFFIX = ".events.npz"

_SCHEMA = artifact_schema("step_events")
#: Each ledger's fields, named as the runtime's C structures name them.
_COLUMNS = {
    "runtime": (
        ("sequence", np.uint64),
        ("timestamp_ns", np.uint64),
        ("step_id", np.uint64),
        ("task_id", np.uint64),
        ("object_id", np.uint64),
        ("allocation_id", np.uint64),
        ("bytes", np.uint64),
        ("kind", np.uint8),
        ("detail_0", np.uint64),
        ("detail_1", np.uint64),
        ("lane_issued_at_ns", np.uint64),
        ("lane_started_at_ns", np.uint64),
        ("lane_finished_at_ns", np.uint64),
    ),
    "allocator": (
        ("sequence", np.uint64),
        ("task_id", np.uint64),
        ("allocation_id", np.uint64),
        ("generation", np.uint64),
        ("requested_bytes", np.uint64),
        ("charged_bytes", np.uint64),
        ("alignment_bytes", np.uint64),
        ("slab_offset", np.uint64),
        ("kind", np.uint8),
        ("category", np.uint8),
    ),
}
#: The runtime's "none" for every field that can lack a value.
_ABSENT = {
    "task_id": NO_ID,
    "object_id": NO_ID,
    "allocation_id": NO_ID,
    "lane_issued_at_ns": NO_STREAM_TIME,
    "lane_started_at_ns": NO_STREAM_TIME,
    "lane_finished_at_ns": NO_STREAM_TIME,
}
#: One record as the decoders read it: the structure's fields by name.
_ROWS = {
    group: namedtuple(f"_{group.title()}Row", [name for name, _ in columns])
    for group, columns in _COLUMNS.items()
}


@dataclass(frozen=True, slots=True)
class StepEvents:
    """A traced step's two event ledgers, decoded as the live trace is."""

    runtime: tuple[RuntimeTraceEvent, ...]
    allocator: tuple[CapturedAllocationEvent, ...]


def events_path(path: Path) -> Path:
    """Where the events of the step file at `path` are kept."""

    return path.with_name(path.stem + EVENTS_SUFFIX)


def write_step_diagnostics(path: Path, diagnostics: StepDiagnostics) -> Path:
    """Write `diagnostics` to `path` and its events beside it, each file whole
    or not at all, and return where the events went."""

    arrays: dict[str, np.ndarray] = {
        "meta": np.frombuffer(json.dumps({"schema": _SCHEMA}).encode(), dtype=np.uint8)
    }
    ledgers = {
        "runtime": diagnostics.runtime.events,
        "allocator": diagnostics.allocator.events,
    }
    for group, records in ledgers.items():
        for name, dtype in _COLUMNS[group]:
            arrays[f"{group}.{name}"] = _column(records, name, dtype)
    events = events_path(path)
    atomic_write(events, lambda output: np.savez(output, allow_pickle=False, **arrays))
    atomic_json(path, diagnostics.as_dict(events=False))
    return events


def read_step_events(path: Path) -> StepEvents:
    """The events kept beside the step file at `path`."""

    source = events_path(path)
    with np.load(source, allow_pickle=False) as data:
        meta = json.loads(data["meta"].tobytes())
        if not isinstance(meta, dict) or meta.get("schema") != _SCHEMA:
            raise ValueError(f"{source}: unsupported step events schema")
        rows = {group: _rows(data, source, group) for group in _COLUMNS}
    try:
        return StepEvents(
            runtime=decode_runtime_events(rows["runtime"]),
            allocator=decode_allocation_events(rows["allocator"]),
        )
    except RuntimeError as exc:
        raise ValueError(f"{source}: {exc}") from exc


def _column(records: Sequence[object], name: str, dtype: type) -> np.ndarray:
    absent = _ABSENT.get(name)
    values = (getattr(item, name) for item in records)
    return np.fromiter(
        (absent if value is None else int(value) for value in values),
        dtype=dtype,
        count=len(records),
    )


def _rows(data: np.lib.npyio.NpzFile, source: Path, group: str) -> list[Any]:
    # The dynamically named rows expose the raw C fields to both decoders.
    columns: list[list[int]] = []
    for name, dtype in _COLUMNS[group]:
        column = data[f"{group}.{name}"]
        if column.ndim != 1 or column.dtype != dtype:
            raise ValueError(
                f"{source}: {group} column {name} is not one row of {np.dtype(dtype)}"
            )
        columns.append(column.tolist())
    if len({len(values) for values in columns}) > 1:
        raise ValueError(f"{source}: the {group} columns differ in length")
    row = _ROWS[group]
    return [row(*fields) for fields in zip(*columns, strict=True)]


__all__ = [
    "EVENTS_SUFFIX",
    "StepEvents",
    "events_path",
    "read_step_events",
    "write_step_diagnostics",
]
