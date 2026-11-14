"""Immutable JSON settings and explicit dataclass record serialization."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Any, Self, cast

type Value = bool | int | float | str | tuple[Value, ...] | Mapping[str, Value] | None


def freeze(value: object, *, name: str = "configuration") -> Value:
    """Own a JSON-compatible tree without importing or encoding model objects."""
    if value is None or type(value) in (bool, int, str):
        return cast(Value, value)
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{name} must contain only finite numbers")
        return value
    if isinstance(value, Mapping):
        if any(type(key) is not str or not key for key in value):
            raise ValueError(f"{name} keys must be nonempty strings")
        return MappingProxyType(
            {key: freeze(item, name=f"{name}.{key}") for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(freeze(item, name=name) for item in value)
    raise TypeError(f"{name} must contain JSON values, not {type(value).__name__}")


def mapping(value: object, name: str) -> Mapping[str, Value]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return cast(Mapping[str, Value], freeze(value, name=name))


def text(value: object, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty string without surrounding spaces")
    return value


def integer(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def number(value: object, name: str, *, minimum: float | None = None) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(cast(int | float, value))
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def names(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of names")
    result = tuple(text(item, name) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def plain(value: object) -> Any:
    """Return owned ordinary containers suitable for JSON serialization."""
    if isinstance(value, Record):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return value


@dataclass(frozen=True)
class Record:
    """A validated settings record; decoding never imports a named Python type."""

    def to_dict(self) -> dict[str, Any]:
        return {item.name: plain(getattr(self, item.name)) for item in fields(self)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        if not isinstance(value, Mapping):
            raise TypeError(f"{cls.__name__} requires an object")
        known = {item.name for item in fields(cls)}
        if unknown := value.keys() - known:
            raise ValueError(f"{cls.__name__}: unknown fields {sorted(unknown)}")
        constructor = cast(Callable[..., Self], cls)
        return constructor(**value)


def record[T: Record](cls: type[T], value: object, name: str) -> T:
    if isinstance(value, cls):
        return value
    if isinstance(value, Mapping):
        return cls.from_dict(value)
    raise TypeError(f"{name} must be {cls.__name__}")
