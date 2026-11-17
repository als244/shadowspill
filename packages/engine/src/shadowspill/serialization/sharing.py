"""Reuse immutable records only within one owned shared-metadata load."""

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, cast


class _Scope:
    def __init__(self, values: Sequence[Any]) -> None:
        # Retain the inputs until all readers finish; object IDs cannot be reused.
        self.values = values
        self.identities = {id(value) for value in values if type(value) in (dict, list)}
        self.records: dict[tuple[object, int], Any] = {}


_current: ContextVar[_Scope | None] = ContextVar("artifact_read_scope", default=None)


@contextmanager
def shared_metadata(values: Sequence[Any]) -> Iterator[None]:
    """The archive owns these read-only input containers for this decode only."""
    token = _current.set(_Scope(values))
    try:
        yield
    finally:
        _current.reset(token)


def share[T](namespace: object, value: Any, build: Callable[[], T]) -> T:
    scope = _current.get()
    if scope is None or id(value) not in scope.identities:
        return build()
    key = namespace, id(value)
    if key not in scope.records:
        scope.records[key] = build()
    return cast(T, scope.records[key])


def reading_shared(value: object) -> bool:
    scope = _current.get()
    return scope is not None and id(value) in scope.identities


def read_record[T](reader: Callable[[object, str], T], value: object, path: str) -> T:
    """Read an immutable record once; its diagnostic path is not its identity."""
    return share(reader, value, lambda: reader(value, path))
