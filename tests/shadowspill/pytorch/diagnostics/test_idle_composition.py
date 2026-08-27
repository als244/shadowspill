"""Compute-stream idle is exactly waiting plus not having been reached.

The summary used to report one number for both, and the two move
independently: a schedule that fetches later raises the waiting without
touching the dispatch cost. Reading the total alone says the boundary got
more expensive when it did not.
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from shadowspill.pytorch.diagnostics.collection import _idle_composition
from shadowspill.pytorch.diagnostics.execution import TaskExecutionTiming


def _task(
    ordinal: int,
    *,
    reached: float,
    started: float,
    finished: float,
) -> TaskExecutionTiming:
    """One task placed on the compute-stream timeline by its three markers.

    Only the markers matter here, so everything else is filled from the
    dataclass itself rather than listed -- a new dispatch field should not
    need a line in this file to keep the composition tested.
    """

    stated: dict[str, object] = {
        "task_id": f"task_{ordinal:06d}",
        "execution_ordinal": ordinal,
        "execution_task_id": f"execution_{ordinal:06d}",
        "semantic_name": f"stage_{ordinal}",
        "phase": "forward",
        "microbatch": 0,
        "expected_profile_seconds": finished - started,
        "gpu_start_seconds": started,
        "gpu_end_seconds": finished,
        "compute_duration_seconds": finished - started,
        "compute_reached_seconds": reached,
        "compute_started_seconds": started,
        "compute_finished_seconds": finished,
        "readiness_wait_seconds": started - reached,
        "compute_reached_sequence": ordinal * 3 + 1,
        "compute_started_sequence": ordinal * 3 + 2,
        "compute_finished_sequence": ordinal * 3 + 3,
    }
    return TaskExecutionTiming(
        **{
            field.name: stated.get(field.name, 0.0)
            for field in fields(TaskExecutionTiming)
        }  # type: ignore[arg-type]
    )


def test_idle_is_exactly_waiting_plus_not_yet_reached() -> None:
    """The two parts account for the span's idle with nothing left over."""

    tasks = (
        # opens with a long fetch, then computes for 1.0
        _task(0, reached=0.0, started=2.0, finished=3.0),
        # reached 0.5 after the previous ended, then waited 0.25 for inputs
        _task(1, reached=3.5, started=3.75, finished=4.75),
        # reached promptly, nothing to wait for
        _task(2, reached=4.8, started=4.8, finished=5.8),
    )
    readiness, dispatch, initial = _idle_composition(tasks)

    assert readiness == pytest.approx(0.25)
    assert dispatch == pytest.approx(0.5 + 0.05)
    assert initial == pytest.approx(2.0)

    span = tasks[-1].compute_finished_seconds - tasks[0].compute_started_seconds
    busy = sum(item.compute_duration_seconds for item in tasks)
    assert readiness + dispatch == pytest.approx(span - busy)


def test_the_first_wait_is_reported_apart_from_the_span() -> None:
    """It ends where the span starts, so no span-relative number holds it."""

    tasks = (
        _task(0, reached=0.0, started=9.0, finished=10.0),
        _task(1, reached=10.0, started=10.0, finished=11.0),
    )
    readiness, dispatch, initial = _idle_composition(tasks)

    assert initial == pytest.approx(9.0)
    assert readiness == pytest.approx(0.0)
    assert dispatch == pytest.approx(0.0)


def test_no_tasks_compose_to_nothing() -> None:
    assert _idle_composition(()) == (0.0, 0.0, 0.0)
