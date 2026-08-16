"""Shared strict JSON primitives for reusable planning artifacts."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _mapping(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{path}: expected an object")
    return value


def _list(value: object, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{path}: expected a list")
    return value


def _string(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{path}: expected a string")
    return value


def _optional_string(value: object, path: str) -> str | None:
    if value is None:
        return None
    return _string(value, path)


def _integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path}: expected an integer")
    return value


def _optional_integer(value: object, path: str) -> int | None:
    if value is None:
        return None
    return _integer(value, path)


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path}: expected a boolean")
    return value


def _pair(value: object, path: str) -> list[Any]:
    result = _list(value, path)
    if len(result) != 2:
        raise ValueError(f"{path}: expected exactly two values")
    return result


def _string_tuple(value: object, path: str) -> tuple[str, ...]:
    return tuple(
        _string(item, f"{path}[{index}]")
        for index, item in enumerate(_list(value, path))
    )


def _integer_pairs(value: object, path: str) -> tuple[tuple[str, int], ...]:
    result: list[tuple[str, int]] = []
    for index, raw in enumerate(_list(value, path)):
        pair = _list(raw, f"{path}[{index}]")
        if len(pair) != 2:
            raise ValueError(f"{path}[{index}]: expected a pair")
        result.append(
            (
                _string(pair[0], f"{path}[{index}][0]"),
                _integer(pair[1], f"{path}[{index}][1]"),
            )
        )
    return tuple(result)
