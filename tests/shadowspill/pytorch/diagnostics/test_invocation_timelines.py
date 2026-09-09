"""The step is the compute stream's cycle: origin to next origin, every invocation."""

from __future__ import annotations

import pytest

from shadowspill.pytorch.diagnostics.timing import InvocationTimelines


class _Clock:
    """A device clock the fake events read when recorded."""

    def __init__(self) -> None:
        self.now_ms = 0.0

    def advance(self, milliseconds: float) -> None:
        self.now_ms += milliseconds


class _Event:
    """A timing event: records the clock, measures against another record."""

    def __init__(self, clock: _Clock) -> None:
        self._clock = clock
        self.recorded_ms: float | None = None
        self.synchronized = 0

    def record(self, stream: object) -> None:
        self.recorded_ms = self._clock.now_ms

    def synchronize(self) -> None:
        self.synchronized += 1

    def elapsed_time(self, other: _Event) -> float:
        assert self.recorded_ms is not None and other.recorded_ms is not None
        return other.recorded_ms - self.recorded_ms


def _timelines(clock: _Clock, capacity: int = 16) -> InvocationTimelines:
    return InvocationTimelines(lambda: _Event(clock), capacity=capacity)


def _invoke(timelines: InvocationTimelines, clock: _Clock, step: int) -> None:
    timelines.begin(step, stream=None)
    clock.advance(5.0)  # the head: the first task's readiness wait
    timelines.start_span(None)
    clock.advance(100.0)  # the tasks
    timelines.end_span(None)
    clock.advance(2.0)  # terminal work still on the stream


def test_a_cycle_closes_when_the_next_invocation_begins() -> None:
    clock = _Clock()
    timelines = _timelines(clock)
    _invoke(timelines, clock, 1)
    assert timelines.drain() == ()  # nothing has followed the first invocation
    _invoke(timelines, clock, 2)
    (first,) = timelines.drain()
    assert first.step_number == 1
    assert first.cycle_seconds == pytest.approx(0.107)
    assert first.head_wait_seconds == pytest.approx(0.005)
    assert first.selected_span_seconds == pytest.approx(0.100)
    assert first.exposed_tail_seconds == pytest.approx(0.002)
    assert first.cycle_seconds == pytest.approx(
        first.head_wait_seconds
        + first.selected_span_seconds
        + first.exposed_tail_seconds
    )
    assert timelines.drain() == ()  # once each


def test_the_end_marker_closes_the_last_cycle_where_the_next_origin_would() -> None:
    clock = _Clock()
    timelines = _timelines(clock)
    _invoke(timelines, clock, 1)
    timelines.mark_end(None)
    (only,) = timelines.drain()
    assert only.step_number == 1
    assert only.cycle_seconds == pytest.approx(0.107)
    # a second marker changes nothing, and the next invocation starts afresh
    timelines.mark_end(None)
    _invoke(timelines, clock, 2)
    assert timelines.drain() == ()
    timelines.mark_end(None)
    (second,) = timelines.drain()
    assert second.step_number == 2


def test_reading_waits_only_for_the_closing_event() -> None:
    clock = _Clock()
    timelines = _timelines(clock)
    _invoke(timelines, clock, 1)
    _invoke(timelines, clock, 2)
    (first,) = timelines.drain()
    second_origin = timelines.current
    assert second_origin is not None and second_origin.origin.synchronized == 1
    assert first.step_number == 1


def test_timelines_are_reused_only_after_being_drained() -> None:
    clock = _Clock()
    timelines = _timelines(clock, capacity=3)
    for step in range(1, 4):
        _invoke(timelines, clock, step)
    # three timelines exist, two cycles are complete and the third is running
    assert len(timelines.drain()) == 2
    _invoke(timelines, clock, 4)
    assert [item.step_number for item in timelines.drain()] == [3]
    timelines.mark_end(None)
    assert [item.step_number for item in timelines.drain()] == [4]


def test_an_undrained_loop_loses_only_the_oldest_completed_cycles() -> None:
    clock = _Clock()
    timelines = _timelines(clock, capacity=3)
    for step in range(1, 7):
        _invoke(timelines, clock, step)
    timelines.mark_end(None)
    steps = [item.step_number for item in timelines.drain()]
    assert steps == [4, 5, 6]


def test_capacity_below_two_is_refused() -> None:
    with pytest.raises(ValueError):
        _timelines(_Clock(), capacity=1)
