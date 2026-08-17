"""Deterministic public-pytree path handling for shared declarations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

type PathComponent = str | int
type PytreePath = tuple[PathComponent, ...]


def normalize_path(path: Sequence[PathComponent]) -> PytreePath:
    """Validate and freeze one public input or output path."""

    result = tuple(path)
    for index, component in enumerate(result):
        if isinstance(component, bool) or not isinstance(component, (str, int)):
            raise TypeError(
                f"pytree path component {index} must be str or int, "
                f"not {type(component).__name__}"
            )
        if isinstance(component, int) and component < 0:
            raise ValueError("pytree path indices must be non-negative")
        if isinstance(component, str) and not component:
            raise ValueError("pytree path names must be non-empty")
    return result


def resolve_path(value: object, path: PytreePath) -> object:
    """Resolve a declaration path without flatten-order assumptions."""

    current = value
    traversed: list[PathComponent] = []
    for component in path:
        traversed.append(component)
        location = format_path(tuple(traversed))
        if isinstance(component, int):
            if not isinstance(current, Sequence) or isinstance(
                current, (str, bytes, bytearray)
            ):
                raise KeyError(f"{location} indexes a non-sequence value")
            try:
                current = current[component]
            except IndexError as error:
                raise KeyError(f"{location} is outside the sequence") from error
            continue
        if isinstance(current, Mapping):
            try:
                current = current[component]
            except KeyError as error:
                raise KeyError(f"{location} is not present in the mapping") from error
            continue
        try:
            current = getattr(current, component)
        except AttributeError as error:
            raise KeyError(f"{location} is not present on the value") from error
    return current


def format_path(path: PytreePath) -> str:
    """Return a stable human-readable path for diagnostics."""

    if not path:
        return "$"
    result = "$"
    for component in path:
        if isinstance(component, int):
            result += f"[{component}]"
        elif component.isidentifier():
            result += f".{component}"
        else:
            result += f"[{component!r}]"
    return result


__all__ = [
    "PathComponent",
    "PytreePath",
    "format_path",
    "normalize_path",
    "resolve_path",
]
