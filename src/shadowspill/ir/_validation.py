"""Shared validation primitives for immutable IR records."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Never


class ValidationError(ValueError):
    """A field-addressable IR validation failure."""

    def __init__(self, path: str, message: str) -> None:
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}")


def fail(path: str, message: str) -> Never:
    raise ValidationError(path, message)


def require(condition: bool, path: str, message: str) -> None:
    if not condition:
        fail(path, message)


def require_identifier(value: str, path: str) -> None:
    require(isinstance(value, str), path, "must be a string")
    require(bool(value), path, "must not be empty")
    require(value.strip() == value, path, "must not have surrounding whitespace")


def require_non_negative(value: int, path: str) -> None:
    require(not isinstance(value, bool), path, "must be an integer")
    require(isinstance(value, int), path, "must be an integer")
    require(value >= 0, path, "must be non-negative")


def require_positive(value: int, path: str) -> None:
    require_non_negative(value, path)
    require(value > 0, path, "must be positive")


def require_tuple(value: object, path: str) -> None:
    require(isinstance(value, tuple), path, "must be a tuple")


def index_unique(values: Iterable[str], path: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for index, value in enumerate(values):
        require_identifier(value, f"{path}[{index}]")
        if value in result:
            fail(
                f"{path}[{index}]",
                f"duplicates {value!r} first seen at index {result[value]}",
            )
        result[value] = index
    return result


def expect_mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        fail(path, "must be an object")
    for key in value:
        if not isinstance(key, str):
            fail(path, "must have string keys")
    return value


def field(mapping: Mapping[str, object], key: str, path: str) -> object:
    if key not in mapping:
        fail(f"{path}.{key}", "is required")
    return mapping[key]


def expect_string(value: object, path: str) -> str:
    if not isinstance(value, str):
        fail(path, "must be a string")
    return value


def expect_integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        fail(path, "must be an integer")
    return value


def expect_boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        fail(path, "must be a boolean")
    return value


def expect_list(value: object, path: str) -> list[object]:
    if not isinstance(value, list):
        fail(path, "must be an array")
    return value
