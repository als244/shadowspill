"""Bounded canonical JSON envelopes. No Python object decoding or tensor JSON."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from .._records import Record, integer
from ..config import (
    BatchConfig,
    Execution,
    Memory,
    Model,
    Objective,
    Optimizer,
    Preparation,
)
from .controls import ControlSchema
from .definitions import DefinitionSource, OutputSet
from .inputs import InputSchema
from .requests import ComputeRequest
from .results import Acceptance, ComputeResult, Failure, RequestStatus

VERSION = 1
MAX_METADATA_BYTES = 1024 * 1024
_KINDS: Mapping[str, type[Record]] = {
    "config.model": Model,
    "config.objective": Objective,
    "config.optimizer": Optimizer,
    "config.batch": BatchConfig,
    "config.memory": Memory,
    "config.execution": Execution,
    "config.preparation": Preparation,
    "schema.inputs": InputSchema,
    "schema.controls": ControlSchema,
    "schema.outputs": OutputSet,
    "definition.source": DefinitionSource,
    "compute.request": ComputeRequest,
    "request.acceptance": Acceptance,
    "compute.result": ComputeResult,
    "request.failure": Failure,
    "request.status": RequestStatus,
}
_NAMES = {value: key for key, value in _KINDS.items()}


class ProtocolError(ValueError):
    """Unsupported version or malformed bounded protocol metadata."""


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ProtocolError(f"nonfinite JSON number: {value}")


def _depth(value: Any) -> None:
    pending = [(value, 0)]
    while pending:
        current, depth = pending.pop()
        if depth > 64:
            raise ProtocolError("metadata nesting exceeds 64 levels")
        if isinstance(current, dict):
            pending.extend((v, depth + 1) for v in current.values())
        elif isinstance(current, list):
            pending.extend((v, depth + 1) for v in current)


def encode(value: Record, *, max_bytes: int = MAX_METADATA_BYTES) -> bytes:
    integer(max_bytes, "max_bytes", minimum=1)
    kind = _NAMES.get(type(value))
    if kind is None:
        raise ProtocolError(f"unregistered protocol record: {type(value).__name__}")
    try:
        payload = value.to_dict()
        _depth(payload)
        encoded = json.dumps(
            {"version": VERSION, "kind": kind, "value": payload},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as error:
        raise ProtocolError(f"invalid {kind} metadata: {error}") from error
    if len(encoded) > max_bytes:
        raise ProtocolError(
            f"metadata exceeds {max_bytes} bytes; transfer tensor payloads separately"
        )
    return encoded


def decode(data: bytes, *, max_bytes: int = MAX_METADATA_BYTES) -> Record:
    integer(max_bytes, "max_bytes", minimum=1)
    if not isinstance(data, bytes):
        raise TypeError("protocol metadata must be bytes")
    if len(data) > max_bytes:
        raise ProtocolError(f"metadata exceeds {max_bytes} bytes")
    try:
        envelope = json.loads(data, object_pairs_hook=_object, parse_constant=_constant)
        _depth(envelope)
        if not isinstance(envelope, dict) or set(envelope) != {
            "version",
            "kind",
            "value",
        }:
            raise ProtocolError("expected version, kind and value envelope fields")
        if type(envelope["version"]) is not int or envelope["version"] != VERSION:
            raise ProtocolError(
                f"unsupported protocol version {envelope['version']!r}; "
                f"expected {VERSION}"
            )
        if not isinstance(envelope["kind"], str) or envelope["kind"] not in _KINDS:
            raise ProtocolError(f"unknown protocol record kind: {envelope['kind']!r}")
        return _KINDS[envelope["kind"]].from_dict(envelope["value"])
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        if isinstance(error, ProtocolError):
            raise
        raise ProtocolError(f"invalid protocol metadata: {error}") from error


def digest(value: Record) -> str:
    """SHA-256 of canonical versioned metadata, as in engine artifact manifests.

    Request metadata includes immutable payload checksums, so the identity covers
    binary data without expanding it or hashing the tensors a second time.
    """
    return hashlib.sha256(encode(value)).hexdigest()
