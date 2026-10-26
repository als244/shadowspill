"""Default-off task ranges shared by saved and older execution paths."""

from contextlib import AbstractContextManager, nullcontext

from .bridge import RuntimeBridge
from .report import profile_range_begin, profile_range_end, set_profiler_annotations


class TaskBoundaryAnnotations:
    """Own the one annotation policy shared by every task executor."""

    __slots__ = ("_bridge", "_disabled_range", "enabled")

    def __init__(self, bridge: RuntimeBridge) -> None:
        self._bridge = bridge
        self._disabled_range = nullcontext()
        self.enabled = False

    def set_enabled(self, enabled: bool) -> None:
        set_profiler_annotations(self._bridge, enabled)
        self.enabled = enabled

    def range(self, name: str) -> AbstractContextManager[None]:
        """Return a zero-work range while annotations are disabled."""

        if not self.enabled:
            return self._disabled_range
        return _EnabledRange(self._bridge, name)

    def begin(self, name: str) -> int:
        """Open one range on an already-checked enabled path."""

        return profile_range_begin(self._bridge, name)

    def end(self, range_id: int) -> None:
        """Close one range opened by :meth:`begin`."""

        profile_range_end(self._bridge, range_id)


class _EnabledRange(AbstractContextManager[None]):
    """Open one provider range only on the explicitly enabled cold path."""

    __slots__ = ("_bridge", "_name", "_range_id")

    def __init__(self, bridge: RuntimeBridge, name: str) -> None:
        self._bridge = bridge
        self._name = name
        self._range_id = 0

    def __enter__(self) -> None:
        self._range_id = profile_range_begin(self._bridge, self._name)

    def __exit__(self, *exc: object) -> None:
        profile_range_end(self._bridge, self._range_id)
