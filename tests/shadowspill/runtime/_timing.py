"""A fake runtime library for the timing markers, driven by a clock a test sets.

`Marker` is a handle on an event the runtime owns, so a test that wants to
control what the device did stands in for the library rather than for the
marker. Every call the real library answers is answered here against
`Clock.now_ms`, which the test advances.
"""

from __future__ import annotations

from typing import Any


class Clock:
    """The device clock a recorded marker reads."""

    def __init__(self) -> None:
        self.now_ms = 0.0

    def advance(self, milliseconds: float) -> None:
        self.now_ms += milliseconds


class TimingLibrary:
    """The timing entry points of `shadowspill.runtime.abi.runtime_library()`."""

    def __init__(self, clock: Clock | None = None, *, tick_ms: float = 0.0) -> None:
        self.clock = clock if clock is not None else Clock()
        #: Advanced before each record, so consecutive spans are `tick_ms`
        #: apart and a caller sampling until its readings settle terminates.
        self.tick_ms = tick_ms
        self.waits = 0
        self.stream_waits = 0
        self.released: list[int] = []
        self._recorded: dict[int, float] = {}
        self._next_marker = 1

    # Every marker is reached the moment it is recorded, which is what a test
    # about bookkeeping wants; a test about pending work clears the entry.
    def pending(self, marker: int) -> None:
        self._recorded.pop(marker, None)

    def shadowspill_timing_marker_create(self, runtime_handle: Any, out: Any) -> int:
        del runtime_handle
        out._obj.value = self._next_marker
        self._next_marker += 1
        return 0

    def shadowspill_timing_marker_record(self, marker: Any, stream: Any) -> int:
        del stream
        self.clock.advance(self.tick_ms)
        self._recorded[int(marker.value)] = self.clock.now_ms
        return 0

    def shadowspill_timing_marker_query(self, marker: Any, reached: Any) -> int:
        reached._obj.value = 1 if int(marker.value) in self._recorded else 0
        return 0

    def shadowspill_timing_marker_wait(self, marker: Any) -> int:
        del marker
        self.waits += 1
        return 0

    def shadowspill_timing_stream_wait(self, runtime_handle: Any, stream: Any) -> int:
        del runtime_handle, stream
        self.stream_waits += 1
        return 0

    def shadowspill_timing_elapsed(
        self, earlier: Any, later: Any, reached: Any, nanoseconds: Any
    ) -> int:
        first = self._recorded.get(int(earlier.value))
        second = self._recorded.get(int(later.value))
        if first is None or second is None:
            reached._obj.value = 0
            return 0
        reached._obj.value = 1
        nanoseconds._obj.value = max(0, round((second - first) * 1e6))
        return 0

    def shadowspill_timing_marker_release(self, marker: Any) -> None:
        self.released.append(int(marker.value))
        self._recorded.pop(int(marker.value), None)


def install(monkeypatch: Any, library: TimingLibrary) -> TimingLibrary:
    """Answer every `Marker` call from ``library`` for the duration of a test."""

    monkeypatch.setattr("shadowspill.runtime.timing.runtime_library", lambda: library)
    return library


__all__ = ["Clock", "TimingLibrary", "install"]
