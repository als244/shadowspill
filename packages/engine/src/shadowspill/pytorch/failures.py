"""Typed failure inspection shared by local and distributed preparation."""

from __future__ import annotations

from collections.abc import Iterator

from torch import OutOfMemoryError


def exception_chain(error: BaseException) -> Iterator[BaseException]:
    """Visit the error and its chained causes once, preserving wrapper errors."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or (
            None if current.__suppress_context__ else current.__context__
        )


def device_exhausted(error: BaseException) -> bool:
    """Recognize a typed allocation failure through planning/profiling wrappers."""
    return any(isinstance(link, OutOfMemoryError) for link in exception_chain(error))
