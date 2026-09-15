"""What the capture reports its phases to."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Protocol


class PhaseTimer(Protocol):
    """What the capture reports its phases to: `measure(name)` brackets one."""

    def measure(self, name: str) -> AbstractContextManager[None]: ...


class NoTimer:
    def measure(self, name: str) -> AbstractContextManager[None]:
        return nullcontext()


__all__ = ["NoTimer", "PhaseTimer"]
