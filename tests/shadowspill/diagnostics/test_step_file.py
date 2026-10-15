"""A step file holds the diagnostics; its events, beside it, read back exactly."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from shadowspill.diagnostics.step import StepDiagnostics
from shadowspill.diagnostics.step_file import (
    StepEvents,
    events_path,
    read_step_events,
    write_step_diagnostics,
)
from shadowspill.runtime.telemetry import (
    AllocationCategory,
    AllocationEventKind,
    CapturedAllocationEvent,
)
from shadowspill.runtime.trace import RuntimeTraceEvent, RuntimeTraceEventKind
from tests.shadowspill.pytorch.diagnostics.test_step_diagnostics_serialization import (
    _diagnostics,
)

_RUNTIME_EVENTS = (
    RuntimeTraceEvent(
        sequence=0,
        timestamp_ns=10,
        step_id=1,
        task_id=None,
        object_id=None,
        allocation_id=None,
        bytes=0,
        kind=RuntimeTraceEventKind.SESSION_BEGIN,
        detail_0=0,
        detail_1=0,
        lane_issued_at_ns=None,
        lane_started_at_ns=None,
        lane_finished_at_ns=None,
    ),
    RuntimeTraceEvent(
        sequence=1,
        timestamp_ns=20,
        step_id=1,
        task_id=3,
        object_id=7,
        allocation_id=11,
        bytes=1 << 20,
        kind=RuntimeTraceEventKind.TRANSFER_COMPLETED,
        detail_0=0,
        detail_1=2,
        lane_issued_at_ns=12,
        lane_started_at_ns=13,
        lane_finished_at_ns=19,
    ),
)
_ALLOCATION_EVENTS = (
    CapturedAllocationEvent(
        sequence=0,
        task_id=None,
        allocation_id=5,
        generation=1,
        requested_bytes=1000,
        charged_bytes=1024,
        alignment_bytes=256,
        slab_offset=4096,
        kind=AllocationEventKind.CREATED,
        category=AllocationCategory.ANONYMOUS,
    ),
    CapturedAllocationEvent(
        sequence=1,
        task_id=3,
        allocation_id=5,
        generation=1,
        requested_bytes=1000,
        charged_bytes=1024,
        alignment_bytes=256,
        slab_offset=4096,
        kind=AllocationEventKind.LOGICAL_FREED,
        category=AllocationCategory.ANONYMOUS,
    ),
)


def _with_events() -> StepDiagnostics:
    diagnostics = _diagnostics()
    return replace(
        diagnostics,
        runtime=replace(diagnostics.runtime, events=_RUNTIME_EVENTS),
        allocator=replace(diagnostics.allocator, events=_ALLOCATION_EVENTS),
    )


def test_the_step_file_is_every_view_but_the_events(tmp_path: Path) -> None:
    diagnostics = _with_events()
    path = tmp_path / "steps" / "27gib.json"

    events = write_step_diagnostics(path, diagnostics)

    assert events == tmp_path / "steps" / "27gib.events.npz" == events_path(path)
    stored = json.loads(path.read_text())
    assert stored == json.loads(json.dumps(diagnostics.as_dict(events=False)))
    assert "events" not in stored["runtime"]
    assert "events" not in stored["allocator"]
    complete = diagnostics.as_dict()
    assert complete["runtime"]["events"][1]["kind"] == "transfer_completed"


def test_the_events_read_back_as_the_trace_decoded_them(tmp_path: Path) -> None:
    path = tmp_path / "1gib.json"
    write_step_diagnostics(path, _with_events())

    assert read_step_events(path) == StepEvents(_RUNTIME_EVENTS, _ALLOCATION_EVENTS)


def test_a_step_without_events_writes_empty_columns(tmp_path: Path) -> None:
    path = tmp_path / "1gib.json"
    write_step_diagnostics(path, _diagnostics())

    assert read_step_events(path) == StepEvents((), ())


def _rewritten(path: Path, change) -> None:
    with np.load(path, allow_pickle=False) as data:
        members = {name: data[name] for name in data.files}
    change(members)
    np.savez(path, **members)


def test_events_of_another_schema_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "1gib.json"
    write_step_diagnostics(path, _with_events())

    def other_schema(members: dict[str, np.ndarray]) -> None:
        members["meta"] = np.frombuffer(
            json.dumps({"schema": "shadowspill.other/v1"}).encode(), dtype=np.uint8
        )

    _rewritten(events_path(path), other_schema)
    with pytest.raises(ValueError, match="schema"):
        read_step_events(path)


def test_event_columns_of_unequal_length_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "1gib.json"
    write_step_diagnostics(path, _with_events())

    def shorter(members: dict[str, np.ndarray]) -> None:
        members["runtime.bytes"] = members["runtime.bytes"][:-1]

    _rewritten(events_path(path), shorter)
    with pytest.raises(ValueError, match="runtime columns differ in length"):
        read_step_events(path)


def test_an_event_kind_nothing_names_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "1gib.json"
    write_step_diagnostics(path, _with_events())

    def unknown(members: dict[str, np.ndarray]) -> None:
        kinds = members["runtime.kind"].copy()
        kinds[0] = 200
        members["runtime.kind"] = kinds

    _rewritten(events_path(path), unknown)
    with pytest.raises(ValueError, match="unknown event kind"):
        read_step_events(path)
