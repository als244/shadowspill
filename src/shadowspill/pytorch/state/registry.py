"""Runtime-scoped ownership for persistent frontend state."""

from __future__ import annotations

import threading
import weakref

from shadowspill.runtime import Runtime
from shadowspill.runtime.objects import (
    release_persistent_state,
    retain_persistent_state,
)

from .records import PersistentState, PersistentStorage


class PersistentStateRegistry:
    """Map public Python objects to the runtime objects that back their state."""

    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime
        self._lock = threading.RLock()
        self._states: dict[int, PersistentState] = {}
        self._storages: dict[int, PersistentStorage] = {}
        self._runtime_objects: dict[int, PersistentStorage] = {}

    def runtime_object(self, identity: int) -> PersistentStorage | None:
        """Find a persistent representation by the runtime's object identity."""
        with self._lock:
            item = self._runtime_objects.get(identity)
            if item is not None and item.current_object_id == identity:
                return item
            # Reindex a legacy rekey once; normal plan adoption keeps IDs stable.
            for item in self._storages.values():
                if item.current_object_id == identity:
                    self._runtime_objects[identity] = item
                    return item
            return None

    def storage(self, identity: int) -> PersistentStorage | None:
        """Find the authoritative object behind a frontend storage view."""
        with self._lock:
            return self._storages.get(identity)

    def get(self, target: object) -> PersistentState | None:
        with self._lock:
            state = self._states.get(id(target))
            if state is not None and state.target is not target:
                raise RuntimeError("persistent state identity was unexpectedly reused")
            return state

    def add(
        self,
        state: PersistentState,
        *,
        allow_in_progress_plan: bool = False,
    ) -> None:
        with self._lock:
            key = id(state.target)
            if key in self._states:
                raise RuntimeError("state is already persistent in this Runtime")
            retain_persistent_state(
                self.runtime, allow_in_progress_plan=allow_in_progress_plan
            )
            self._states[key] = state
            self._storages.update(
                (item.storage_identity, item) for item in state.storages
            )
            self._runtime_objects.update(
                (item.current_object_id, item) for item in state.storages
            )

    def remove(self, target: object) -> PersistentState:
        with self._lock:
            state = self._states.pop(id(target), None)
            if state is None or state.target is not target:
                raise RuntimeError("state is not persistent in this Runtime")
            release_persistent_state(self.runtime)
            for item in state.storages:
                self._storages.pop(item.storage_identity, None)
            removed = {id(item) for item in state.storages}
            self._runtime_objects = {
                key: item
                for key, item in self._runtime_objects.items()
                if id(item) not in removed
            }
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
