"""Runtime-owned logical object references.

The neutral runtime owns byte ranges, generations, readiness, and residency.
Framework frontends may layer shape or type information on an ``ObjectRef``,
but this module deliberately has no framework or device-backend dependency.
"""

from __future__ import annotations

import threading
from typing import Protocol, runtime_checkable


@runtime_checkable
class ObjectReferenceOwner(Protocol):
    """Private ownership operation required by :class:`ObjectRef`."""

    def _release_object_reference(self, reference: ObjectRef) -> None: ...


class ObjectRef:
    """One open reference to a runtime object, wherever its bytes are.

    The runtime makes one when it retains an object for a caller, and does not
    close while any of its references is open. The object's identity outlasts
    any one lease: each generation of its value may have leases in any of the
    runtime's pools, and a plan's pool and route bindings decide where it must
    be. As a context manager, it closes on exit.
    """

    __slots__ = (
        "_closed",
        "_handle",
        "_lock",
        "_object_id",
        "_owner",
        "_size_bytes",
    )

    def __init__(
        self,
        owner: ObjectReferenceOwner,
        *,
        object_id: int,
        size_bytes: int,
        handle: int,
    ) -> None:
        if object_id < 0:
            raise ValueError("runtime object ID must be non-negative")
        if size_bytes < 0:
            raise ValueError("runtime object size must be non-negative")
        if handle <= 0:
            raise ValueError("runtime object handle must be positive")
        self._owner = owner
        self._object_id = object_id
        self._size_bytes = size_bytes
        self._handle = handle
        self._lock = threading.RLock()
        self._closed = False

    @property
    def object_id(self) -> int:
        """The object's identity, unique within the runtime."""

        return self._object_id

    @property
    def size_bytes(self) -> int:
        """The extent of the object's storage, in bytes."""

        return self._size_bytes

    @property
    def closed(self) -> bool:
        """Whether this reference has been closed."""

        with self._lock:
            return self._closed

    def close(self) -> None:
        """Release this reference; closing it again does nothing.

        Raises
        ------
        RuntimeExecutionError
            If the runtime refuses the release. The reference stays open.
        """

        with self._lock:
            if self._closed:
                return
            self._owner._release_object_reference(self)
            self._closed = True

    def _require_open(self) -> None:
        if self.closed:
            raise RuntimeError("runtime object reference is closed")

    def _belongs_to(self, owner: object) -> bool:
        """Return whether ``owner`` created this reference."""

        return self._owner is owner

    def _require_handle(self) -> int:
        """Return the runtime handle after checking this reference is open."""

        self._require_open()
        return self._handle

    def __enter__(self) -> ObjectRef:
        self._require_open()
        return self

    def __exit__(self, *exception: object) -> None:
        del exception
        self.close()

    def __repr__(self) -> str:
        state = "closed" if self.closed else "open"
        return (
            f"ObjectRef(object_id={self.object_id}, "
            f"size_bytes={self.size_bytes}, {state})"
        )


__all__ = ["ObjectRef", "ObjectReferenceOwner"]
