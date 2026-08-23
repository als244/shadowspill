"""Small immutable and picklable mapping used by public diagnostics."""

from __future__ import annotations

from collections.abc import Iterator, Mapping


class FrozenMapping[Key, Value](Mapping[Key, Value]):
    """Insertion-ordered immutable mapping with a stable pickle form.

    ``types.MappingProxyType`` is immutable but cannot be pickled. Diagnostics
    are keyed by execution identity on common inspection paths, so this wrapper
    retains ``dict`` lookup cost without exposing any mutation operation.
    """

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[Key, Value]) -> None:
        self._values = dict(values)

    def __getitem__(self, key: Key) -> Value:
        return self._values[key]

    def __iter__(self) -> Iterator[Key]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __reduce__(self) -> tuple[object, tuple[dict[Key, Value]]]:
        return type(self), (self._values,)


__all__ = ["FrozenMapping"]
