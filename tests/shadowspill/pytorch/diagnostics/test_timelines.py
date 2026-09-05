"""Transfer lanes join simulated intervals with stream stamps on one timeline."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from shadowspill.planner.diagnostics.mapping import FrozenMapping
from shadowspill.pytorch.diagnostics.collection import (
    _lane_summary,
    _object_relations,
    _transfer_record,
)
from shadowspill.pytorch.diagnostics.execution import TransferRecord
from shadowspill.pytorch.runtime_adapter.trace import (
    RuntimeTraceEvent,
    RuntimeTraceEventKind,
)
from shadowspill.simulator import TransferDirection, TransferInterval

_ORIGIN_NS = 1_000_000_000


def _event(
    kind: RuntimeTraceEventKind,
    *,
    timestamp_ns: int,
    stream: tuple[int, int] | None = None,
) -> RuntimeTraceEvent:
    return RuntimeTraceEvent(
        sequence=0,
        timestamp_ns=timestamp_ns,
        step_id=1,
        task_id=3,
        object_id=7,
        allocation_id=None,
        bytes=1 << 20,
        kind=kind,
        detail_0=0,
        detail_1=0,
        lane_started_at_ns=None if stream is None else stream[0],
        lane_finished_at_ns=None if stream is None else stream[1],
    )


def _interval(sequence: int, *, start_ns: int, end_ns: int) -> TransferInterval:
    return TransferInterval(
        alias_group_id="alias_000007",
        trigger_task_id="task_000003",
        device_id="cuda_0",
        direction=TransferDirection.FETCH,
        sequence=sequence,
        ready_ns=start_ns - 1_000,
        start_ns=start_ns,
        end_ns=end_ns,
        bytes=1 << 20,
    )


def _record(
    sequence: int,
    *,
    simulated: tuple[int, int],
    stream: tuple[int, int] | None,
    simulated_origin_ns: int = 0,
    alignment: float = 0.0,
) -> TransferRecord:
    dispatch = _event(
        RuntimeTraceEventKind.TRANSFER_DISPATCHED, timestamp_ns=_ORIGIN_NS + 5_000
    )
    completion = _event(
        RuntimeTraceEventKind.TRANSFER_COMPLETED,
        timestamp_ns=_ORIGIN_NS + 9_000,
        stream=stream,
    )
    return _transfer_record(
        _interval(sequence, start_ns=simulated[0], end_ns=simulated[1]),
        "fetch",
        "execution_000003",
        ("execution_000002", "execution_000004", "execution_000001"),
        dispatch,
        completion,
        None,
        None,
        simulated_origin_ns,
        alignment,
        _ORIGIN_NS,
    )


def test_deltas_are_taken_after_aligning_the_simulation() -> None:
    """A transfer that ran exactly when simulated has zero deltas.

    The simulation counts from the first selected task's start; the device
    counts from the origin event. With the first task starting 0.25 s after
    the origin, a simulated start of 1.0 s is a device time of 1.25 s.
    """

    record = _record(
        0,
        simulated=(1_000_000_000, 1_010_000_000),
        stream=(1_250_000_000, 1_260_000_000),
        alignment=0.25,
    )
    assert record.simulated_started_at_seconds == pytest.approx(1.0)
    assert record.lane_started_at_seconds == pytest.approx(1.25)
    assert record.start_delta_seconds == pytest.approx(0.0)
    assert record.end_delta_seconds == pytest.approx(0.0)
    assert (
        record.lane_finished_at_seconds
        - record.lane_started_at_seconds
        - (record.simulated_finished_at_seconds - record.simulated_started_at_seconds)
    ) == pytest.approx(0.0)
    assert record.dispatched_at_seconds == pytest.approx(5e-6)
    assert record.completion_observed_at_seconds == pytest.approx(9e-6)
    assert record.previous_access == "execution_000002"
    assert record.next_access == "execution_000004"
    assert record.modified_by == "execution_000001"


def test_an_unmeasured_transfer_keeps_its_host_times_and_no_deltas() -> None:
    record = _record(0, simulated=(0, 10_000_000), stream=None)
    assert record.lane_started_at_seconds is None
    assert record.lane_finished_at_seconds is None
    assert record.start_delta_seconds is None
    assert record.dispatched_at_seconds == pytest.approx(5e-6)


def test_lane_summary_reports_measured_bandwidth_and_the_largest_drift() -> None:
    measured = _record(
        0, simulated=(0, 10_000_000), stream=(0, 20_000_000)
    )  # 1 MiB in 20 ms: half the assumed speed, and it ran 0 s late
    late = _record(
        1, simulated=(100_000_000, 110_000_000), stream=(130_000_000, 140_000_000)
    )  # ran 30 ms late
    unmeasured = _record(2, simulated=(200_000_000, 210_000_000), stream=None)
    summary = _lane_summary("fetch", (measured, late, unmeasured), ())
    assert summary.transfers == 3
    assert summary.measured_transfers == 2
    assert summary.bytes == 3 << 20
    assert summary.simulated_busy_seconds == pytest.approx(0.03)
    assert summary.lane_busy_seconds == pytest.approx(0.03)
    assert summary.effective_bandwidth_bytes_per_second == pytest.approx(
        (2 << 20) / 0.03
    )
    assert summary.largest_start_delta_seconds == pytest.approx(0.03)
    assert summary.largest_start_delta_transfer_id == "fetch_000001"
    assert summary.opening_transfers == 0


def test_object_relations_name_the_neighbours_and_the_modifier() -> None:
    """Accesses are relative to the trigger; the modifier is the last write."""

    timing = SimpleNamespace(
        tasks={"task_000003": SimpleNamespace(execution_ordinal=3)},
        alias_accesses=FrozenMapping(
            {
                "written": ((1, True), (2, False), (3, True), (5, False), (6, False)),
                "input": ((0, False), (4, False)),
                "unused": ((1, True),),
            }
        ),
    )
    assert _object_relations(timing, "written", "task_000003") == (  # type: ignore[arg-type]
        "execution_000003",
        "execution_000005",
        "execution_000003",
    )
    assert _object_relations(timing, "input", "task_000003") == (  # type: ignore[arg-type]
        "execution_000000",
        "execution_000004",
        "init",
    )
    assert _object_relations(timing, "unused", "task_000003") == (  # type: ignore[arg-type]
        "execution_000001",
        "persistent",
        "execution_000001",
    )
    assert _object_relations(timing, "absent", "task_000003") == (  # type: ignore[arg-type]
        "init",
        "persistent",
        "init",
    )
