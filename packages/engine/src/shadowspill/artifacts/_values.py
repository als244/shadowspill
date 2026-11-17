"""Small JSON-only value helpers shared by the persistent artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from shadowspill.serialization.sharing import share


def freeze(value: Any) -> Any:
    # Artifact metadata contains millions of scalar leaves. Avoid Mapping's
    # abstract-class checks for ordinary JSON values; keep the general path for
    # subclasses and custom mappings, including their existing validation.
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    return share(freeze, value, lambda: _freeze(value))


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("metadata keys must be strings")
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError(f"metadata must contain finite JSON values, got {type(value)}")


def plain(value: Any) -> Any:
    if value is None or type(value) in (str, bool, int, float):
        return value
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def encoded(value: Any) -> bytes:
    return json.dumps(
        value,
        default=_json_mapping,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _json_mapping(value: Any) -> dict[Any, Any]:
    # The JSON encoder already handles scalar leaves, dicts, lists and tuples.
    # Only immutable/custom mappings need conversion. Avoid making another full
    # Python copy of a large Program before every integrity hash.
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def digest(value: Any) -> str:
    return hashlib.sha256(encoded(value)).hexdigest()


def nonnegative_int(value: int, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def seconds(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
