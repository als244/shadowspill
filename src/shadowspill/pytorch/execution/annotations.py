"""Shared, default-off profiler ranges for frontend task boundaries."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from shadowspill.pytorch.runtime_adapter.bridge import RuntimeBridge


class TaskBoundaryAnnotations:
    """Own the one annotation policy shared by every task executor."""

    __slots__ = ("_bridge", "enabled")

    def __init__(self, bridge: RuntimeBridge) -> None:
        self._bridge = bridge
        self.enabled = False

    def set_enabled(self, enabled: bool) -> None:
        self._bridge.set_profiler_annotations(enabled)
        self.enabled = enabled

    @contextmanager
    def range(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        range_id = self._bridge.profile_range_begin(name)
        try:
            yield
        finally:
            self._bridge.profile_range_end(range_id)


__all__ = ["TaskBoundaryAnnotations"]
