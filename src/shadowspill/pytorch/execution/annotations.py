"""Shared, default-off profiler ranges for frontend task boundaries."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext

from shadowspill.pytorch.runtime_adapter.bridge import RuntimeBridge


class TaskBoundaryAnnotations:
    """Own the one annotation policy shared by every task executor."""

    __slots__ = ("_bridge", "_disabled_range", "enabled")

    def __init__(self, bridge: RuntimeBridge) -> None:
        self._bridge = bridge
        self._disabled_range = nullcontext()
        self.enabled = False

    def set_enabled(self, enabled: bool) -> None:
        self._bridge.set_profiler_annotations(enabled)
        self.enabled = enabled

    def range(self, name: str) -> AbstractContextManager[None]:
        """Return a zero-work problem while annotations are disabled."""

        if not self.enabled:
            return self._disabled_range
        return _EnabledRange(self._bridge, name)

    def begin(self, name: str) -> int:
        """Open one range on an already-checked enabled path."""

        return self._bridge.profile_range_begin(name)

    def end(self, range_id: int) -> None:
        """Close one range opened by :meth:`begin`."""

        self._bridge.profile_range_end(range_id)


class _EnabledRange(AbstractContextManager[None]):
    """Open one provider range only on the explicitly enabled cold path."""

    __slots__ = ("_bridge", "_name", "_range_id")

    def __init__(self, bridge: RuntimeBridge, name: str) -> None:
        self._bridge = bridge
        self._name = name
        self._range_id = 0

    def __enter__(self) -> None:
        self._range_id = self._bridge.profile_range_begin(self._name)

    def __exit__(self, *exc: object) -> None:
        self._bridge.profile_range_end(self._range_id)


__all__ = ["TaskBoundaryAnnotations"]
