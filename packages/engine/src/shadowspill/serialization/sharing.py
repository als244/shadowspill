"""Reuse records and metadata only within one owned archive read or write."""

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any, cast


class _Scope:
    def __init__(self, values: Sequence[Any]) -> None:
        # Retain the inputs until all readers finish; object IDs cannot be reused.
        self.values = values
        self.identities = {id(value) for value in values if type(value) in (dict, list)}
        self.records: dict[tuple[object, int], Any] = {}


_current: ContextVar[_Scope | None] = ContextVar("artifact_read_scope", default=None)
_writing: ContextVar[dict[tuple[object, int], tuple[Any, Any]] | None] = ContextVar(
    "artifact_write_scope", default=None
)


@contextmanager
def writing_metadata() -> Iterator[None]:
    """Keep shared inputs alive and convert them once during this archive write."""
    token = _writing.set({})
    try:
        yield
    finally:
        _writing.reset(token)


def write_shared[T](namespace: object, value: Any, build: Callable[[], T]) -> T:
    """Preserve sharing while writing; ordinary ``to_dict`` owns fresh values."""
    scope = _writing.get()
    if scope is None:
        return build()
    key = namespace, id(value)
    found = scope.get(key)
    if found is None:
        result = build()
        # Retaining the input prevents object-ID reuse by temporary records.
        scope[key] = value, result
        return result
    return cast(T, found[1])


def shared_writer[T, R](writer: Callable[[T], R]) -> Callable[[T], R]:
    """Convert an immutable record once per archive, with no cross-call cache."""

    @wraps(writer)
    def write(value: T) -> R:
        return write_shared(writer, value, lambda: writer(value))

    return write


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
