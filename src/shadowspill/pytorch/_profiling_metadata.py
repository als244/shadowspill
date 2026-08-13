"""Canonical planning-only metadata for value-sensitive task profiling."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

_SCHEMA = "shadowspill.profiling_metadata/v1"


@dataclass(frozen=True, slots=True)
class ProfilingMetadata:
    """Canonical JSON and digest for one workload class."""

    canonical_json: str
    digest: str


def canonicalize_profiling_metadata(value: object) -> ProfilingMetadata:
    normalized = _normalize(value, "profiling_metadata")
    payload = {"schema": _SCHEMA, "value": normalized}
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return ProfilingMetadata(
        encoded,
        hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
    )


def training_profiling_metadata(
    value: Sequence[object] | None,
    *,
    microbatch_count: int,
) -> tuple[ProfilingMetadata, ...]:
    if value is None:
        return tuple(
            canonicalize_profiling_metadata(None) for _ in range(microbatch_count)
        )
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("profiling_metadata must be a sequence or None")
    if len(value) != microbatch_count:
        raise ValueError(
            "profiling_metadata must have one entry per example microbatch"
        )
    return tuple(canonicalize_profiling_metadata(item) for item in value)


def _normalize(value: object, path: str) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} floating values must be finite")
        return value
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError(f"{path} mapping keys must be non-empty strings")
            result[key] = _normalize(item, f"{path}.{key}")
        return {key: result[key] for key in sorted(result)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            _normalize(item, f"{path}[{index}]") for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{path} must contain only JSON-compatible scalars, sequences, and mappings"
    )


__all__ = [
    "ProfilingMetadata",
    "canonicalize_profiling_metadata",
    "training_profiling_metadata",
]
