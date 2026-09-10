"""Shared strict JSON primitives for reusable planning artifacts."""

from __future__ import annotations

import hashlib
import json

from ..strict import (
    _boolean,
    _integer,
    _integer_pairs,
    _list,
    _mapping,
    _optional_integer,
    _optional_string,
    _pair,
    _string,
    _string_tuple,
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


__all__ = [
    "_boolean",
    "_canonical_json",
    "_digest",
    "_integer",
    "_integer_pairs",
    "_list",
    "_mapping",
    "_optional_integer",
    "_optional_string",
    "_pair",
    "_string",
    "_string_tuple",
]
