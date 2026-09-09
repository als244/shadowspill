"""Runtime-scoped ownership for persistent frontend state."""

from __future__ import annotations

import threading
import weakref

from shadowspill.pytorch.runtime_adapter.runtime import Runtime

from .records import PersistentState, PersistentStorage


class PersistentStateRegistry:
    """Map public Python objects to the runtime objects that back their state."""

    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime
        self._lock = threading.RLock()
        self._states: dict[int, PersistentState] = {}
        #: Pool memory handed to a caller that no state owns yet, by the
        #: identity of the storage presenting it. An import consults this to
        #: adopt what is already in the pool instead of copying the pool into
        #: itself under a second name, so no caller has to say so.
        self._pool_allocations: dict[int, PersistentStorage] = {}

    def note_pool_allocation(self, allocation: PersistentStorage) -> None:
        """Record pool memory presented as a host storage."""

        with self._lock:
            self._pool_allocations[allocation.storage_identity] = allocation

    def pool_allocation(self, identity: int) -> PersistentStorage | None:
        """The pool allocation this storage presents, if it is one."""

        with self._lock:
            return self._pool_allocations.get(identity)

    def forget_pool_allocation(self, identity: int) -> None:
        """Stop offering an allocation, once a state owns it or it is given back."""

        with self._lock:
            self._pool_allocations.pop(identity, None)

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
            self.runtime._retain_persistent_state(
                allow_in_progress_plan=allow_in_progress_plan
            )
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
