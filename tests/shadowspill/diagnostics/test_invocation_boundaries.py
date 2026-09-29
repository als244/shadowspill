"""Entry and terminal transfers share the invocation's unchanged time axis."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from shadowspill.diagnostics.collection import _build_step_summary, _transfer_lanes
from shadowspill.runtime.trace import RuntimeTraceEventKind

from .test_idle_composition import _task
from .test_timelines import _ORIGIN_NS, _event, _interval, _record


def test_summary_components_include_entry_and_terminal_once() -> None:
    tasks = (
        replace(
            _task(1, reached=0.01, started=0.05, finished=0.16),
            expected_profile_seconds=0.1,
        ),
        replace(
            _task(2, reached=0.17, started=0.20, finished=0.43),
            expected_profile_seconds=0.2,
        ),
    )
    intervals = {
        "task_000000": SimpleNamespace(start_ns=0, end_ns=0, stall_ns=0),
        "task_000001": SimpleNamespace(
            start_ns=20_000_000, end_ns=120_000_000, stall_ns=20_000_000
        ),
        "task_000002": SimpleNamespace(
            start_ns=150_000_000, end_ns=350_000_000, stall_ns=30_000_000
        ),
    }
    lanes = {
        "evict": SimpleNamespace(
            records=(
                _record(0, simulated=(0, 400_000_000), stream=(0, 500_000_000)),
                _record(1, simulated=(0, 80_000_000), stream=(0, 80_000_000)),
            )
        )
    }
    timing = SimpleNamespace(
        timeline=None,
        dispatch_call_started_ns=0,
        dispatch_call_finished_ns=1,
        prior_invocation_drain_ns=0,
        trace_setup_ns=0,
    )
    runtime = SimpleNamespace(event_overflow=False, allocation_event_overflow=False)
    summary = _build_step_summary(
        timing,
        SimpleNamespace(makespan_ns=400_000_000),
        tasks,
        intervals,
        runtime,
        lanes,
    )  # type: ignore[arg-type]
    assert summary.simulated_entry_delay_seconds == pytest.approx(0.02)
    assert summary.simulated_inter_task_readiness_wait_seconds == pytest.approx(0.03)
    assert summary.entry_delay_seconds == pytest.approx(0.05)
    assert summary.real_terminal_tail_seconds == pytest.approx(0.07)
    assert summary.real_invocation_seconds == pytest.approx(0.50)
    assert sum(
        (
            summary.entry_delay_seconds,
            summary.real_task_event_seconds,
            summary.real_inter_task_idle_seconds,
            summary.real_terminal_tail_seconds,
        )
    ) == pytest.approx(0.50)
    assert sum(
        (
            summary.simulated_entry_delay_seconds,
            summary.profiled_task_seconds,
            summary.simulated_inter_task_idle_seconds,
            summary.simulator_terminal_tail_seconds,
        )
    ) == pytest.approx(0.40)
    assert {item.phase for item in summary.phase_comparisons} == {"forward"}

    lanes["evict"].records = (_record(0, simulated=(0, 400_000_000), stream=None),)
    incomplete = _build_step_summary(
        timing,
        SimpleNamespace(makespan_ns=400_000_000),
        tasks,
        intervals,
        runtime,
        lanes,
    )  # type: ignore[arg-type]
    assert incomplete.real_invocation_seconds is None
    assert incomplete.real_terminal_tail_seconds is None
    assert not incomplete.trace_complete


@pytest.mark.parametrize("extra", [False, True])
def test_every_runtime_transfer_must_match_the_schedule(extra: bool) -> None:
    dispatch = _event(
        RuntimeTraceEventKind.TRANSFER_DISPATCHED, timestamp_ns=_ORIGIN_NS
    )
    completion = _event(
        RuntimeTraceEventKind.TRANSFER_COMPLETED,
        timestamp_ns=_ORIGIN_NS + 10,
        stream=(0, 10),
    )
    events = (dispatch, completion, dispatch) if extra else (dispatch, completion)
    timing = SimpleNamespace(
        task_order=("task_000003",),
        tasks={"task_000003": SimpleNamespace(execution_ordinal=0)},
        alias_accesses={},
    )
    simulation = SimpleNamespace(
        transfer_intervals=(_interval(0, start_ns=0, end_ns=10),)
    )
    evidence = SimpleNamespace(runtime_trace=SimpleNamespace(events=events))
    bridge = SimpleNamespace(
        objects=SimpleNamespace(runtime_object_id=lambda alias: 7),
        runtime=SimpleNamespace(routes={}),
        spill_pool_id=1,
        execution_pool_id=0,
    )
    if extra:
        with pytest.raises(RuntimeError, match="transfer count mismatch"):
            _transfer_lanes(timing, simulation, evidence, bridge, _ORIGIN_NS)  # type: ignore[arg-type]
    else:
        lanes = _transfer_lanes(timing, simulation, evidence, bridge, _ORIGIN_NS)  # type: ignore[arg-type]
        (record,) = lanes["fetch"].records
        assert record.triggered_by == "execution_000000"
        assert record.start_delta_seconds == 0.0
        assert lanes["fetch"].summary.transfers == 1
