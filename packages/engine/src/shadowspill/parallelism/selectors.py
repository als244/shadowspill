"""Resolve names once, including ties, without ordered-rule precedence."""

from collections.abc import Mapping, Sequence
from fnmatch import fnmatchcase


def select[T](names: Sequence[str], entries: Mapping[str, T], default: T) -> T:
    exact = [entries[name] for name in names if name in entries]
    values = exact or [
        value
        for pattern, value in entries.items()
        if any(fnmatchcase(name, pattern) for name in names)
    ]
    if not values:
        return default
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"conflicting declarations for tied names {tuple(names)}")
    return values[0]


def validate_selectors(names: Sequence[str], entries: Mapping[str, object]) -> None:
    for pattern in entries:
        if not any(fnmatchcase(name, pattern) for name in names):
            raise ValueError(
                f"parameter selector matches no registered tensor: {pattern}"
            )
