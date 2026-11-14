"""Bounded canonical JSON envelopes. No Python object decoding or tensor JSON."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from .._records import Record, integer, text
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
from .creation import TrainingCreation
from .definitions import DefinitionSource, OutputSet
from .inputs import InputSchema
from .preparation import ResolvedPreparation
from .requests import OperationRequest, WriterLease
from .results import (
    Acceptance,
    ExecutionReceipt,
    Failure,
    OperationResult,
    RequestStatus,
)
from .runs import RunInfo
from .service import ServiceError, ServiceInfo
from .training import ResolvedTraining

VERSION = 1
MAX_METADATA_BYTES = 1024 * 1024
MAX_METADATA_NODES = 16384
_KINDS: Mapping[str, type[Record]] = {
    "config.model": Model,
    "config.objective": Objective,
    "config.optimizer": Optimizer,
    "config.batch": BatchConfig,
    "config.memory": Memory,
    "config.execution": Execution,
    "config.preparation": Preparation,
    "config.resolved_preparation": ResolvedPreparation,
    "config.resolved_training": ResolvedTraining,
    "schema.inputs": InputSchema,
    "schema.controls": ControlSchema,
    "schema.outputs": OutputSet,
    "definition.source": DefinitionSource,
    "compute.request": OperationRequest,
    "writer.lease": WriterLease,
    "request.acceptance": Acceptance,
    "compute.result": OperationResult,
    "compute.execution": ExecutionReceipt,
    "request.failure": Failure,
    "request.status": RequestStatus,
    "service.info": ServiceInfo,
    "service.error": ServiceError,
    "run.info": RunInfo,
    "training.creation": TrainingCreation,
}


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
    nodes = 0
    while pending:
        current, depth = pending.pop()
        nodes += 1
        if depth > 64:
            raise ProtocolError("metadata nesting exceeds 64 levels")
        if isinstance(current, dict):
            nodes += len(current)  # Field names are decoded string objects too.
        children = len(current) if isinstance(current, dict | list) else 0
        if nodes + len(pending) + children > MAX_METADATA_NODES:
            raise ProtocolError(f"metadata exceeds {MAX_METADATA_NODES} nodes")
        if isinstance(current, dict):
            pending.extend((v, depth + 1) for v in current.values())
        elif isinstance(current, list):
            pending.extend((v, depth + 1) for v in current)


def _structure(data: bytes) -> None:
    """Bound JSON object growth before allocation; json.loads still validates it.

    A small byte string can otherwise create hundreds of thousands of tiny
    containers. Count containers, scalar values and field names without storing
    tokens. Escaped quotes and structural characters inside strings are inert.
    """
    nodes = depth = 0
    quoted = escaped = primitive = False
    for char in data:
        if quoted:
            if escaped:
                escaped = False
            elif char == 92:
                escaped = True
            elif char == 34:
                quoted = False
            continue
        if char == 34:
            nodes += 1
            quoted, primitive = True, False
        elif char in (91, 123):
            nodes += 1
            depth += 1
            primitive = False
            if depth > 65:
                raise ProtocolError("metadata nesting exceeds 64 levels")
        elif char in (93, 125):
            depth -= 1
            primitive = False
        elif char in (9, 10, 13, 32, 44, 58):
            primitive = False
        elif not primitive:
            nodes += 1
            primitive = True
        if nodes > MAX_METADATA_NODES:
            raise ProtocolError(f"metadata exceeds {MAX_METADATA_NODES} nodes")


class RecordCodec:
    """A fixed, explicit record allowlist using the common bounded JSON format.

    Protocols can own independent registries without adding private records to
    the public client protocol or importing classes named by incoming messages.
    """

    def __init__(self, kinds: Mapping[str, type[Record]]) -> None:
        if not kinds:
            raise ValueError("a protocol codec requires at least one record kind")
        for kind, cls in kinds.items():
            text(kind, "record kind")
            if not isinstance(cls, type) or not issubclass(cls, Record):
                raise TypeError("protocol record kinds must be Record classes")
        if len(set(kinds.values())) != len(kinds):
            raise ValueError("a record class cannot have multiple protocol kinds")
        self._kinds = MappingProxyType(dict(kinds))
        self._names = MappingProxyType({value: key for key, value in kinds.items()})

    def encode(self, value: Record, *, max_bytes: int = MAX_METADATA_BYTES) -> bytes:
        integer(max_bytes, "max_bytes", minimum=1)
        kind = self._names.get(type(value))
        if kind is None:
            raise ProtocolError(f"unregistered protocol record: {type(value).__name__}")
        try:
            payload = value.to_dict()
            envelope = {"version": VERSION, "kind": kind, "value": payload}
            _depth(envelope)
            encoded = json.dumps(
                envelope,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError, RecursionError) as error:
            raise ProtocolError(f"invalid {kind} metadata: {error}") from error
        if len(encoded) > max_bytes:
            raise ProtocolError(
                f"metadata exceeds {max_bytes} bytes; "
                "transfer tensor payloads separately"
            )
        return encoded

    def decode(self, data: bytes, *, max_bytes: int = MAX_METADATA_BYTES) -> Record:
        integer(max_bytes, "max_bytes", minimum=1)
        if not isinstance(data, bytes):
            raise TypeError("protocol metadata must be bytes")
        if len(data) > max_bytes:
            raise ProtocolError(f"metadata exceeds {max_bytes} bytes")
        envelope = None
        try:
            _structure(data)
            envelope = json.loads(
                data, object_pairs_hook=_object, parse_constant=_constant
            )
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
            if (
                not isinstance(envelope["kind"], str)
                or envelope["kind"] not in self._kinds
            ):
                raise ProtocolError(
                    f"unknown protocol record kind: {envelope['kind']!r}"
                )
            return self._kinds[envelope["kind"]].from_dict(envelope["value"])
        except (TypeError, ValueError, UnicodeError, RecursionError) as error:
            message = str(error)[:1024]
            if not isinstance(error, ProtocolError):
                message = "invalid protocol metadata: " + message
        finally:
            # Retained exceptions must not retain the input or partial decoded
            # graph after the caller releases its metadata reservation.
            data, envelope = b"", None
        # Raise outside the handler so neither JSONDecodeError.doc nor a failed
        # record constructor's traceback becomes this exception's context.
        raise ProtocolError(message)


_DEFAULT = RecordCodec(_KINDS)


def encode(value: Record, *, max_bytes: int = MAX_METADATA_BYTES) -> bytes:
    return _DEFAULT.encode(value, max_bytes=max_bytes)


def decode(data: bytes, *, max_bytes: int = MAX_METADATA_BYTES) -> Record:
    try:
        return _DEFAULT.decode(data, max_bytes=max_bytes)
    finally:
        data = b""


def digest(value: Record) -> str:
    """SHA-256 of canonical versioned metadata, as in engine artifact manifests.

    Request metadata includes immutable payload checksums, so the identity covers
    binary data without expanding it or hashing the tensors a second time.
    """
    return hashlib.sha256(encode(value)).hexdigest()
