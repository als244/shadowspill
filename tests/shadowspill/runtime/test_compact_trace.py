"""Compact snapshots preserve evidence after buffer reuse without eager records."""

from __future__ import annotations

import ctypes
import json
import pickle
from dataclasses import asdict, replace
from pathlib import Path

import pytest

import shadowspill.runtime.trace as trace
from shadowspill.diagnostics.collection.transfers import transfer_lanes
from shadowspill.diagnostics.step_file import read_step_events, write_step_diagnostics
from shadowspill.runtime._event_records import PackedEventRecords
from shadowspill.runtime.abi import AllocationEvent, TraceEvent, TraceSummary
from shadowspill.runtime.telemetry import (
    NO_ID,
    AllocationCategory,
    AllocationEventKind,
    AllocationTelemetryError,
    decode_allocation_events,
    pack_allocation_events,
)
from shadowspill.runtime.trace import (
    RuntimeTraceEventKind,
    decode_runtime_events,
    pack_runtime_events,
)
from shadowspill.status import ABI_VERSION
from tests.shadowspill.diagnostics.test_invocation_boundaries import _one_fetch
from tests.shadowspill.diagnostics.test_step_serialization import _diagnostics
from tests.shadowspill.diagnostics.test_timelines import _ORIGIN_NS, _event


def _buffers():
    runtime = (TraceEvent * len(RuntimeTraceEventKind))()
    allocations = (AllocationEvent * len(AllocationEventKind))()
    for index, event in enumerate(runtime):
        event.sequence = index
        event.timestamp_ns = (1 << 63) + index
        event.step_id = 19
        event.task_id = NO_ID if index == 0 else 3
        event.object_id = NO_ID if index == 0 else 7
        event.allocation_id = NO_ID if index == 0 else 11
        event.kind = index
        event.bytes = 1024
        event.detail_0, event.detail_1 = 1, 23
        event.lane_issued_at_ns = NO_ID if index == 0 else 100
        event.lane_started_at_ns = NO_ID if index == 0 else 200
        event.lane_finished_at_ns = NO_ID if index == 0 else 300
    for index, event in enumerate(allocations):
        event.sequence = index
        event.pool_id = 2
        event.task_id = NO_ID if index == 0 else 3
        event.allocation_id = 11
        event.generation = 4
        event.requested_bytes, event.charged_bytes = 1000, 1024
        event.alignment_bytes, event.slab_offset = 256, 4096
        event.kind = index
        event.category = index % len(AllocationCategory)
    return runtime, allocations


class _TraceLibrary:
    def __init__(self):
        self.runtime, self.allocations = _buffers()

    def shadowspill_trace_read(self, handle, output, events, capacity, allocations, n):
        summary = TraceSummary(
            abi_version=ABI_VERSION,
            step_id=19,
            event_count=len(self.runtime),
            allocation_event_count=len(self.allocations),
            event_capacity=len(self.runtime) + 100,
            allocation_event_capacity=len(self.allocations) + 100,
            began_at_ns=10,
            ended_at_ns=20,
            origin_host_ns=11,
            event_overflow=1,
        )
        ctypes.memmove(output, ctypes.byref(summary), ctypes.sizeof(summary))
        if events is not None:
            assert capacity == len(self.runtime) and n == len(self.allocations)
            ctypes.memmove(events, self.runtime, ctypes.sizeof(self.runtime))
            ctypes.memmove(
                allocations, self.allocations, ctypes.sizeof(self.allocations)
            )
        return 0


def test_native_copy_owns_all_fields_and_defers_projection(monkeypatch):
    library = _TraceLibrary()
    expected = decode_runtime_events(library.runtime)
    expected_allocations = decode_allocation_events(library.allocations)
    original = trace._decode_runtime_event
    decoded = []

    def record(event):
        decoded.append(event.sequence)
        return original(event)

    monkeypatch.setattr(trace, "runtime_library", lambda: library)
    monkeypatch.setattr(trace, "_decode_runtime_event", record)
    captured = trace.read_runtime_trace(123)
    assert decoded == []
    assert (captured.step_id, captured.origin_host_ns, captured.event_overflow) == (
        19,
        11,
        True,
    )
    ctypes.memset(library.runtime, 0, ctypes.sizeof(library.runtime))
    ctypes.memset(library.allocations, 0, ctypes.sizeof(library.allocations))
    assert captured.events[-1] == expected[-1]
    assert decoded == [len(expected) - 1]
    assert captured.events[1:8:2] == expected[1:8:2]
    assert tuple(captured.events) == expected
    assert tuple(captured.allocation_events) == expected_allocations
    assert captured.events == expected and expected == captured.events
    with pytest.raises(IndexError):
        _ = captured.events[len(expected)]
    with pytest.raises(IndexError):
        _ = captured.events[-len(expected) - 1]
    assert isinstance(captured.events, PackedEventRecords)
    column = captured.events.column("timestamp_ns")
    with pytest.raises(ValueError):
        column[0] = 123
    with pytest.raises(ValueError):
        column.setflags(write=True)


@pytest.mark.parametrize(
    ("group", "field", "value", "message"),
    [
        ("runtime", "sequence", 4, "sequence is not contiguous"),
        ("runtime", "kind", 255, "unknown event kind"),
        ("allocator", "sequence", 4, "sequence is not contiguous"),
        ("allocator", "kind", 255, "unknown enum value"),
        ("allocator", "category", 255, "unknown enum value"),
    ],
)
def test_malformed_events_fail_at_capture_before_any_projection(
    monkeypatch, group, field, value, message
):
    library = _TraceLibrary()
    buffer = library.runtime if group == "runtime" else library.allocations
    setattr(buffer[0], field, value)
    monkeypatch.setattr(trace, "runtime_library", lambda: library)
    error = RuntimeError if group == "runtime" else AllocationTelemetryError
    with pytest.raises(error, match=message):
        trace.read_runtime_trace(123)


def test_compact_export_never_expands_events_and_preserves_columns(
    tmp_path, monkeypatch
):
    runtime, allocations = _buffers()
    expected_runtime = decode_runtime_events(runtime)
    expected_allocations = decode_allocation_events(allocations)
    base = _diagnostics()
    diagnostics = replace(
        base,
        runtime=replace(base.runtime, events=pack_runtime_events(runtime)),
        allocator=replace(base.allocator, events=pack_allocation_events(allocations)),
    )

    def forbid_iteration(self):
        raise AssertionError("compact export must not decode event objects")

    monkeypatch.setattr(PackedEventRecords, "__iter__", forbid_iteration)
    path = Path(tmp_path) / "step.json"
    write_step_diagnostics(path, diagnostics)
    decoded = read_step_events(path)
    assert decoded.runtime == expected_runtime
    assert decoded.allocator == expected_allocations


def test_packed_evidence_can_outlive_its_process_and_handles():
    runtime, allocations = _buffers()
    expected_runtime = decode_runtime_events(runtime)
    expected_allocations = decode_allocation_events(allocations)
    payload = pickle.dumps(
        (pack_runtime_events(runtime), pack_allocation_events(allocations))
    )
    del runtime, allocations
    restored_runtime, restored_allocations = pickle.loads(payload)
    assert tuple(restored_runtime) == expected_runtime
    assert tuple(restored_allocations) == expected_allocations
    assert not restored_runtime.column("kind").flags.writeable


@pytest.mark.parametrize("empty", [False, True])
def test_raw_trace_json_export_preserves_decoded_fields(monkeypatch, empty):
    library = _TraceLibrary()
    if empty:
        library.runtime = (TraceEvent * 0)()
        library.allocations = (AllocationEvent * 0)()
    expected_runtime = decode_runtime_events(library.runtime)
    expected_allocations = decode_allocation_events(library.allocations)
    monkeypatch.setattr(trace, "runtime_library", lambda: library)
    captured = trace.read_runtime_trace(123)
    expected = asdict(
        replace(
            captured, events=expected_runtime, allocation_events=expected_allocations
        )
    )
    serialized = json.dumps(captured.as_dict())
    assert json.loads(serialized) == json.loads(json.dumps(expected))
    assert isinstance(captured.events, PackedEventRecords)
    assert isinstance(captured.allocation_events, PackedEventRecords)


def test_empty_records_remain_empty():
    runtime = pack_runtime_events((TraceEvent * 0)())
    allocations = pack_allocation_events((AllocationEvent * 0)())
    assert runtime == allocations == ()
    assert runtime[:] == ()
    assert not runtime and not allocations
    assert list(runtime.matching("kind", 0)) == []


def test_transfer_analysis_projects_only_relevant_runtime_records(monkeypatch):
    dispatch = _event(
        RuntimeTraceEventKind.TRANSFER_DISPATCHED, timestamp_ns=_ORIGIN_NS
    )
    completion = _event(
        RuntimeTraceEventKind.TRANSFER_COMPLETED,
        timestamp_ns=_ORIGIN_NS + 10,
        stream=(0, 10),
    )
    unrelated = _event(
        RuntimeTraceEventKind.RETIREMENT_COMPLETED, timestamp_ns=_ORIGIN_NS
    )
    events = (unrelated, dispatch, unrelated, completion, unrelated)
    buffer = (TraceEvent * len(events))()
    for index, (destination, source) in enumerate(zip(buffer, events, strict=True)):
        for field, _ in TraceEvent._fields_:
            value = getattr(source, field)
            setattr(destination, field, NO_ID if value is None else value)
        destination.sequence = index
    packed = pack_runtime_events(buffer)

    def forbid_iteration(self):
        raise AssertionError("transfer analysis must not expand the full ledger")

    monkeypatch.setattr(PackedEventRecords, "__iter__", forbid_iteration)
    timing, simulation, evidence, bridge = _one_fetch(packed)
    lanes = transfer_lanes(timing, simulation, evidence, bridge, _ORIGIN_NS)
    (record,) = lanes["fetch"].records
    assert record.start_delta_seconds == 0.0
    assert record.end_delta_seconds == 0.0
    assert lanes["fetch"].summary.transfers == 1
