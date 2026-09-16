"""The step is the compute stream's cycle: origin to next origin, every invocation."""

from __future__ import annotations

import pytest

from shadowspill.diagnostics.timing import InvocationTimelines
from tests.shadowspill.runtime._timing import Clock, TimingLibrary, install

STREAM = 11


def _timelines(
    monkeypatch: pytest.MonkeyPatch, clock: Clock, capacity: int = 16
) -> InvocationTimelines:
    install(monkeypatch, TimingLibrary(clock))
    return InvocationTimelines(0, capacity=capacity)


def _invoke(timelines: InvocationTimelines, clock: Clock, step: int) -> None:
    timelines.begin(step, STREAM)
    clock.advance(5.0)  # the head: the first task's readiness wait
    timelines.start_span(STREAM)
    clock.advance(100.0)  # the tasks
    timelines.end_span(STREAM)
    clock.advance(2.0)  # terminal work still on the stream


def test_a_cycle_closes_when_the_next_invocation_begins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    timelines = _timelines(monkeypatch, clock)
    _invoke(timelines, clock, 1)
    assert timelines.drain() == ()  # nothing has followed the first invocation
    _invoke(timelines, clock, 2)
    (first,) = timelines.drain()
    assert first.step_number == 1
    assert first.cycle_seconds == pytest.approx(0.107)
    assert first.opening_delay_seconds == pytest.approx(0.005)
    assert first.selected_span_seconds == pytest.approx(0.100)
    assert first.exposed_tail_seconds == pytest.approx(0.002)
    assert first.cycle_seconds == pytest.approx(
        first.opening_delay_seconds
        + first.selected_span_seconds
        + first.exposed_tail_seconds
    )
    assert timelines.drain() == ()  # once each


def test_the_end_marker_closes_the_last_cycle_where_the_next_origin_would(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    timelines = _timelines(monkeypatch, clock)
    _invoke(timelines, clock, 1)
    timelines.mark_end(STREAM)
    (only,) = timelines.drain()
    assert only.step_number == 1
    assert only.cycle_seconds == pytest.approx(0.107)
    # a second marker changes nothing, and the next invocation starts afresh
    timelines.mark_end(STREAM)
    _invoke(timelines, clock, 2)
    assert timelines.drain() == ()
    timelines.mark_end(STREAM)
    (second,) = timelines.drain()
    assert second.step_number == 2


def test_reading_waits_only_for_the_closing_instant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    library = install(monkeypatch, TimingLibrary(clock))
    timelines = InvocationTimelines(0)
    _invoke(timelines, clock, 1)
    _invoke(timelines, clock, 2)
    (first,) = timelines.drain()
    assert first.step_number == 1
    # One wait, on the instant that closed the cycle. A caller that closes a
    # cycle reads it in the next breath, which is what discarding a warm step
    # before a measured group depends on; nothing waits for the stream.
    assert library.waits == 1
    assert library.stream_waits == 0


def test_timelines_are_reused_only_after_being_drained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    timelines = _timelines(monkeypatch, clock, capacity=3)
    for step in range(1, 4):
        _invoke(timelines, clock, step)
    # three timelines exist, two cycles are complete and the third is running
    assert len(timelines.drain()) == 2
    _invoke(timelines, clock, 4)
    assert [item.step_number for item in timelines.drain()] == [3]
    timelines.mark_end(STREAM)
    assert [item.step_number for item in timelines.drain()] == [4]


def test_an_undrained_loop_loses_only_the_oldest_completed_cycles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    timelines = _timelines(monkeypatch, clock, capacity=3)
    for step in range(1, 7):
        _invoke(timelines, clock, step)
    timelines.mark_end(STREAM)
    steps = [item.step_number for item in timelines.drain()]
    assert steps == [4, 5, 6]


def test_capacity_below_two_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError):
        _timelines(monkeypatch, Clock(), capacity=1)
