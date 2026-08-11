"""Canonical JSON encoding shared by public IR records."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

type JsonScalar = bool | int | float | str | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]


def canonical_json(value: Mapping[str, JsonValue]) -> str:
    """Return the stable wire representation used for identities and caches."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def digest_json(value: Mapping[str, JsonValue]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def parse_json(payload: str) -> object:
    return json.loads(payload)


def string_list(values: Sequence[str]) -> list[JsonValue]:
    return list(values)
