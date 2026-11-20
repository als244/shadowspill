"""Timing work on the device, against the runtime's own events.

The runtime times its transfers with backend events. These are the same events,
offered to a caller timing anything else on the same stream, so one step's
timeline has one clock.

There is one thing here: a `Marker`, where an instant on a stream is recorded. A
caller takes markers once, records them around whatever it wants to measure --
recording again every step, which is how one marker times the same span each
time -- and asks one how long since another, whether the device has reached it
yet, or to be blocked until it has. An interval is two markers; a completion
check is one.

`nanoseconds_between` is the same span for a caller that has already waited and
knows the answer must be there. A caller that must wait for a whole stream
rather than one instant on it calls `wait_for_stream`.

The compute stream is named by the integer handle its owner already has, and the
runtime wraps it through the backend, which is the only thing that knows how.
"""

from __future__ import annotations

import ctypes

from shadowspill.errors import PlanningError

from .abi import runtime_library


class Marker:
    """An instant on a stream, recorded as often as the caller likes.

    Two markers make an interval; one alone says whether the device has
    reached it.
    Markers use the backend events the runtime times its own transfers with,
    so their instants share one clock with the runtime's trace. When every
    timing event is in use, taking a marker adds one rather than using the
    reserve a prepared trace holds; releasing it gives the event back. As a
    context manager, it releases the marker on exit.

    Parameters
    ----------
    runtime_handle : int
        Handle of the open runtime the marker's event comes from.

    Raises
    ------
    PlanningError
        If the runtime cannot provide an event.
    """

    __slots__ = ("_handle",)

    def __init__(self, runtime_handle: int) -> None:
        handle = ctypes.c_void_p(0)
        self._require(
            runtime_library().shadowspill_timing_marker_create(
                ctypes.c_size_t(runtime_handle), ctypes.byref(handle)
            ),
            "take a timing marker",
        )
        self._handle = handle

    def record(self, compute_stream: int) -> None:
        """Record this marker's instant on a stream, replacing any earlier one.

        Parameters
        ----------
        compute_stream : int
            The stream's integer handle, as its owner names it.

        Raises
        ------
        PlanningError
            If the backend refuses to record, or the marker is released.
        """

        self._require(
            runtime_library().shadowspill_timing_marker_record(
                self._handle, ctypes.c_uint64(compute_stream)
            ),
            "record a timing marker",
        )

    @staticmethod
    def _require(raw_status: object, operation: str) -> None:
        status = int(raw_status)  # type: ignore[call-overload]
        if status != 0:
            raise PlanningError(f"{operation} failed with status {status}")

    def reached(self) -> bool:
        """Check, without waiting, whether the device has reached this instant.

        Returns
        -------
        bool
            True once the device has passed it.

        Raises
        ------
        PlanningError
            If the backend query fails, or the marker is released.
        """

        reached = ctypes.c_uint8(0)
        self._require(
            runtime_library().shadowspill_timing_marker_query(
                self._handle, ctypes.byref(reached)
            ),
            "query a timing marker",
        )
        return bool(reached.value)

    def wait(self) -> None:
        """Block the calling thread until the device reaches this instant.

        Raises
        ------
        PlanningError
            If the wait fails, or the marker is released.
        """

        self._require(
            runtime_library().shadowspill_timing_marker_wait(self._handle),
            "wait for a timing marker",
        )

    def nanoseconds_since(self, earlier: Marker) -> int | None:
        """Measure the time from an earlier marker to this one.

        Parameters
        ----------
        earlier : Marker
            The marker recorded first.

        Returns
        -------
        int or None
            Nanoseconds from ``earlier`` to this marker, or None while the
            device has not reached this one.

        Raises
        ------
        PlanningError
            If the backend cannot measure the interval, or a marker is
            released.
        """

        reached = ctypes.c_uint8(0)
        elapsed = ctypes.c_uint64(0)
        self._require(
            runtime_library().shadowspill_timing_elapsed(
                earlier._handle,
                self._handle,
                ctypes.byref(reached),
                ctypes.byref(elapsed),
            ),
            "measure between timing markers",
        )
        return int(elapsed.value) if reached.value else None

    def release(self) -> None:
        """Give the marker's event back to the runtime.

        Releasing it again does nothing. A released marker can no longer be
        recorded or read.
        """

        if self._handle.value:
            runtime_library().shadowspill_timing_marker_release(self._handle)
            self._handle = ctypes.c_void_p(0)

    def __enter__(self) -> Marker:
        return self

    def __exit__(self, *exception: object) -> None:
        self.release()


def nanoseconds_between(earlier: Marker, later: Marker) -> int:
    """Measure the time between two markers the device has already passed.

    `Marker.nanoseconds_since` answers None while the later instant is still
    ahead of the device, which suits a caller that polls; this is for a caller
    that has already waited.

    Parameters
    ----------
    earlier : Marker
        The marker recorded first.
    later : Marker
        The marker recorded second.

    Returns
    -------
    int
        Nanoseconds from ``earlier`` to ``later``.

    Raises
    ------
    PlanningError
        If the device has not reached ``later``, or the interval cannot be
        measured.
    """

    elapsed = later.nanoseconds_since(earlier)
    if elapsed is None:
        raise PlanningError("a timed span was read before the device reached it")
    return elapsed


def wait_for_stream(runtime_handle: int, compute_stream: int) -> None:
    """Block the calling thread until the device finishes all work on a stream.

    Parameters
    ----------
    runtime_handle : int
        Handle of the open runtime.
    compute_stream : int
        The stream's integer handle, as its owner names it.

    Raises
    ------
    PlanningError
        If the wait fails.
    """

    status = int(
        runtime_library().shadowspill_timing_stream_wait(
            ctypes.c_size_t(runtime_handle), ctypes.c_uint64(compute_stream)
        )
    )
    if status != 0:
        raise PlanningError(f"wait for a stream failed with status {status}")


__all__ = ["Marker", "nanoseconds_between", "wait_for_stream"]
