"""Byte and request limits for owned submission buffers, with backpressure."""

from __future__ import annotations

from threading import Condition
from time import monotonic
from typing import Self

from ._records import integer, number


class BufferBudget:
    def __init__(self, max_bytes: int, max_requests: int = 2) -> None:
        self.max_bytes = integer(max_bytes, "max_bytes", minimum=1)
        self.max_requests = integer(max_requests, "max_requests", minimum=1)
        self._condition = Condition()
        self._bytes = self._requests = 0
        self._closed = False

    @property
    def used(self) -> tuple[int, int]:
        with self._condition:
            return self._bytes, self._requests

    def reserve(self, size: int, *, timeout: float | None = None) -> BufferLease:
        integer(size, "buffer bytes")
        if size > self.max_bytes:
            raise ValueError(
                f"input snapshot needs {size} staging bytes; limit is {self.max_bytes}"
            )
        if timeout is not None:
            number(timeout, "timeout", minimum=0)
        deadline = None if timeout is None else monotonic() + timeout
        with self._condition:
            while True:
                if self._closed:
                    raise RuntimeError("input staging is closed")
                if (
                    self._bytes + size <= self.max_bytes
                    and self._requests < self.max_requests
                ):
                    self._bytes += size
                    self._requests += 1
                    return BufferLease(self, size)
                remaining = None if deadline is None else deadline - monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("timed out waiting for input staging capacity")
                self._condition.wait(remaining)

    def close(self) -> None:
        """Refuse new reservations and wake waiters; existing owners still release."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class BufferLease:
    def __init__(self, budget: BufferBudget, size: int) -> None:
        self._budget = budget
        self._size = size
        self._released = False

    def shrink(self, size: int) -> None:
        integer(size, "retained buffer bytes")
        with self._budget._condition:
            if self._released or size > self._size:
                raise ValueError("a buffer lease can only shrink while owned")
            self._budget._bytes -= self._size - size
            self._size = size
            self._budget._condition.notify_all()

    def close(self) -> None:
        with self._budget._condition:
            if self._released:
                return
            self._budget._bytes -= self._size
            self._budget._requests -= 1
            self._released = True
            self._budget._condition.notify_all()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
