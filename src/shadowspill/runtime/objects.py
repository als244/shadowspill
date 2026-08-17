"""Runtime-owned logical object references.

The neutral runtime owns byte ranges, generations, readiness, and residency.
Framework frontends may layer shape or type information on an ``ObjectRef``,
but this module deliberately has no tensor or device-backend dependency.
"""

from __future__ import annotations

import threading
from enum import StrEnum
from typing import Protocol, runtime_checkable


class ObjectConsistency(StrEnum):
    """Cross-plan value-ordering policy for one object binding."""

    CAUSAL = "causal"
    UNORDERED = "unordered"


@runtime_checkable
class ObjectReferenceOwner(Protocol):
    """Private ownership operation required by :class:`ObjectRef`."""

    def _release_object_reference(self, reference: ObjectRef) -> None: ...


class ObjectRef:
    """One retained reference to a runtime-global logical object.

    Object identity is independent of residency. A generation may have leases
    in any configured pool, and planning resolves required residency through
    the selected plan's pool and route bindings.
    """

    __slots__ = (
        "_closed",
        "_lock",
        "_native_handle",
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
        native_handle: int,
    ) -> None:
        if object_id < 0:
            raise ValueError("runtime object ID must be non-negative")
        if size_bytes < 0:
            raise ValueError("runtime object size must be non-negative")
        if native_handle <= 0:
            raise ValueError("runtime object handle must be positive")
        self._owner = owner
        self._object_id = object_id
        self._size_bytes = size_bytes
        self._native_handle = native_handle
        self._lock = threading.RLock()
        self._closed = False

    @property
    def object_id(self) -> int:
        """Return the runtime-global logical identity."""

        return self._object_id

    @property
    def size_bytes(self) -> int:
        """Return the exact byte extent of the logical storage root."""

        return self._size_bytes

    @property
    def closed(self) -> bool:
        """Whether this public reference has released its ownership."""

        with self._lock:
            return self._closed

    def close(self) -> None:
        """Release this reference exactly once."""

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

    def _handle(self) -> int:
        """Return the private native handle after validating ownership."""

        self._require_open()
        return self._native_handle

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


__all__ = ["ObjectConsistency", "ObjectRef", "ObjectReferenceOwner"]
