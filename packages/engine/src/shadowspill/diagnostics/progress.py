"""Bounded preparation/startup progress, without accelerator synchronization."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import wraps
from threading import RLock
from time import monotonic
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")
_current: ContextVar[Progress | None] = ContextVar("host_progress", default=None)


@dataclass
class PhaseOutcome:
    """A recoverably rejected candidate completed its attempt, not its work."""

    rejected: bool = False


class Progress:
    """A bounded hierarchy of actual host operation boundaries.

    A failed display disables this sink without changing accepted computation.
    Wall timestamps locate events; monotonic durations measure host work. They
    are not GPU kernel timings. No polling thread or synthetic progress events.
    """

    def __init__(self, sink: Callable[[str], None]) -> None:
        self._sink = sink
        self._lock = RLock()
        self._stack: list[tuple[str, float]] = []
        self._started = monotonic()
        self._disabled = False

    def event(self, message: str, *, depth: int | None = None) -> None:
        with self._lock:
            if self._disabled:
                return
            stamp = (
                datetime.now(UTC)
                .isoformat(timespec="milliseconds")
                .replace("+00:00", "Z")
            )
            identity = f"rank={os.environ.get('RANK', '0')} pid={os.getpid()}"
            indent = "  " * (len(self._stack) if depth is None else depth)
            try:
                self._sink(
                    f"{stamp} [{identity} +{monotonic() - self._started:.2f}s] "
                    f"{indent}{message}"
                )
            except Exception:
                self._disabled = True

    @contextmanager
    def phase(self, name: str) -> Iterator[PhaseOutcome]:
        started = monotonic()
        with self._lock:
            if len(self._stack) >= 64:
                raise RuntimeError("progress operation nesting exceeds 64")
            depth = len(self._stack)
            self._stack.append((name, started))
            self.event(f"→ {name}", depth=depth)
        outcome = "✓"
        result = PhaseOutcome()
        try:
            yield result
        except BaseException as error:
            outcome = f"✗ {type(error).__name__}:"
            raise
        finally:
            with self._lock:
                if result.rejected and outcome == "✓":
                    outcome = "⊘ Rejected:"
                self.event(
                    f"{outcome} {name} — {monotonic() - started:.2f}s", depth=depth
                )
                self._stack.pop()

    @contextmanager
    def activate(self) -> Iterator[None]:
        token = _current.set(self)
        try:
            yield
        finally:
            _current.reset(token)


@contextmanager
def phase(name: str) -> Iterator[PhaseOutcome]:
    reporter = _current.get()
    if reporter is None:
        yield PhaseOutcome()
    else:
        with reporter.phase(name) as result:
            yield result


def event(message: str) -> None:
    reporter = _current.get()
    if reporter is not None:
        reporter.event(message)


def timed(name: str) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Observe a host operation only when its caller enabled progress."""

    def decorate(function: Callable[P, T]) -> Callable[P, T]:
        @wraps(function)
        def call(*args: P.args, **kwargs: P.kwargs) -> T:
            with phase(name):
                return function(*args, **kwargs)

        return call

    return decorate


def logged_phase(name: str) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Preparation child entry point: retain progress in its owned stdout log."""

    def decorate(function: Callable[P, T]) -> Callable[P, T]:
        @wraps(function)
        def call(*args: P.args, **kwargs: P.kwargs) -> T:
            reporter = Progress(lambda message: print(message, flush=True))
            with reporter.activate(), reporter.phase(name):
                return function(*args, **kwargs)

        return call

    return decorate
