"""Versioned request metadata. Tensor bytes travel in separate bounded payloads."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, cast

from .._records import Record, integer, mapping, names, number, record, text
from .controls import Scalar

type ComputeOperation = Literal["step", "forward_backward", "optim_step", "forward"]
COMPUTE_OPERATIONS = frozenset({"step", "forward_backward", "optim_step", "forward"})


def identifier(value: object, name: str) -> str:
    value = text(value, name)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", value) is None:
        raise ValueError(f"{name} must be an opaque identifier, not a path")
    return value


def scalars(value: Mapping[str, Scalar], name: str) -> Mapping[str, Scalar]:
    values = mapping(value, name)
    for key, item in values.items():
        if type(item) is not bool:
            number(item, f"{name}.{key}")
    # Validation above rejects all containers and non-scalars.
    return cast(Mapping[str, Scalar], values)


@dataclass(frozen=True)
class ArtifactRef(Record):
    id: str
    sha256: str
    size_bytes: int
    media_type: str = "application/octet-stream"

    def __post_init__(self) -> None:
        identifier(self.id, "artifact ID")
        if (
            not isinstance(self.sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None
        ):
            raise ValueError("artifact SHA-256 must contain 64 lowercase hex digits")
        integer(self.size_bytes, "artifact size")
        text(self.media_type, "media type")


@dataclass(frozen=True)
class RequestIdentity(Record):
    request_id: str
    run_id: str
    sequence: int
    writer_id: str
    writer_epoch: int
    server_incarnation: str

    def __post_init__(self) -> None:
        for name in ("request_id", "run_id", "writer_id", "server_incarnation"):
            identifier(getattr(self, name), name)
        integer(self.sequence, "submission sequence", minimum=1)
        integer(self.writer_epoch, "writer epoch", minimum=1)


@dataclass(frozen=True)
class WriterLease(Record):
    run_id: str
    writer_id: str
    writer_epoch: int
    server_incarnation: str
    next_sequence: int

    def __post_init__(self) -> None:
        for name in ("run_id", "writer_id", "server_incarnation"):
            identifier(getattr(self, name), name)
        integer(self.writer_epoch, "writer epoch", minimum=1)
        integer(self.next_sequence, "next submission sequence", minimum=1)

    def permits(self, identity: RequestIdentity) -> bool:
        return (
            identity.run_id == self.run_id
            and identity.writer_id == self.writer_id
            and identity.writer_epoch == self.writer_epoch
            and identity.server_incarnation == self.server_incarnation
            and identity.sequence >= self.next_sequence
        )


@dataclass(frozen=True)
class OperationRequest(Record):
    identity: RequestIdentity
    operation: ComputeOperation
    payload: ArtifactRef | None = None
    datum_count: int = 0
    optim: Mapping[str, Scalar] = field(default_factory=dict)
    loss_params: Mapping[str, Scalar] = field(default_factory=dict)
    grad_scale: float | None = None
    outputs: str | None = None
    mode: Literal["evaluation", "training"] | None = None
    runtime_trace: bool = False
    dependencies: Sequence[str] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "identity", record(RequestIdentity, self.identity, "identity")
        )
        if self.operation not in COMPUTE_OPERATIONS:
            raise ValueError("unknown compute operation")
        if self.payload is not None:
            object.__setattr__(
                self, "payload", record(ArtifactRef, self.payload, "payload")
            )
        integer(self.datum_count, "datum_count")
        if self.datum_count and self.payload is None:
            raise ValueError("nonempty global batches require a payload reference")
        object.__setattr__(self, "optim", scalars(self.optim, "optim"))
        object.__setattr__(
            self, "loss_params", scalars(self.loss_params, "loss_params")
        )
        if self.operation in ("step", "optim_step"):
            object.__setattr__(
                self,
                "grad_scale",
                number(1 if self.grad_scale is None else self.grad_scale, "grad_scale"),
            )
        elif self.grad_scale is not None or self.optim:
            raise ValueError(
                "optimizer controls and grad_scale require step or optim_step"
            )
        if self.operation == "optim_step" and (
            self.payload is not None
            or self.datum_count
            or self.loss_params
            or self.outputs is not None
            or self.mode is not None
        ):
            raise ValueError(
                "optim_step has no batch, objective controls, outputs or mode"
            )
        if self.operation != "optim_step":
            mode = self.mode or (
                "evaluation" if self.operation == "forward" else "training"
            )
            if mode not in ("evaluation", "training") or (
                self.operation != "forward" and mode != "training"
            ):
                raise ValueError("step and backward require training mode")
            object.__setattr__(self, "mode", mode)
        if self.outputs is not None:
            text(self.outputs, "output set")
        if type(self.runtime_trace) is not bool:
            raise TypeError("runtime_trace must be a boolean")
        dependencies = names(self.dependencies, "dependencies")
        for dependency in dependencies:
            identifier(dependency, "dependency request ID")
        if self.identity.request_id in dependencies:
            raise ValueError("a request cannot depend on itself")
        object.__setattr__(self, "dependencies", dependencies)
