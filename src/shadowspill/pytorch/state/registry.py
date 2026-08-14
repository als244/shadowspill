"""Runtime-scoped ownership for persistent frontend state."""

from __future__ import annotations

import threading
import weakref

from shadowspill.pytorch.runtime_adapter.runtime import Runtime

from .records import PersistentState


class PersistentStateRegistry:
    """Map public Python objects to the runtime objects that back their state."""

    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime
        self._lock = threading.RLock()
        self._states: dict[int, PersistentState] = {}

    def get(self, target: object) -> PersistentState | None:
        with self._lock:
            state = self._states.get(id(target))
            if state is not None and state.target is not target:
                raise RuntimeError("persistent state identity was unexpectedly reused")
            return state

    def add(self, state: PersistentState) -> None:
        with self._lock:
            key = id(state.target)
            if key in self._states:
                raise RuntimeError("state is already persistent in this Runtime")
            self.runtime._retain_persistent_state()
            self._states[key] = state

    def remove(self, target: object) -> PersistentState:
        with self._lock:
            state = self._states.pop(id(target), None)
            if state is None or state.target is not target:
                raise RuntimeError("state is not persistent in this Runtime")
            self.runtime._release_persistent_state()
            return state

    def values(self) -> tuple[PersistentState, ...]:
        """Return a lock-consistent snapshot of all persistent state owners."""

        with self._lock:
            return tuple(self._states.values())


_registries_lock = threading.Lock()
_registries: weakref.WeakKeyDictionary[Runtime, PersistentStateRegistry] = (
    weakref.WeakKeyDictionary()
)


def registry_for(runtime: Runtime) -> PersistentStateRegistry:
    """Return the unique frontend state registry for one Runtime."""

    with _registries_lock:
        registry = _registries.get(runtime)
        if registry is None:
            registry = PersistentStateRegistry(runtime)
            _registries[runtime] = registry
        return registry


__all__ = ["PersistentStateRegistry", "registry_for"]
