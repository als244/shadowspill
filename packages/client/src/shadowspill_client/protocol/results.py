"""Durable outcomes distinguish execution effects from result publication."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

from .._records import Record, integer, names, number, record, text
from .controls import Scalar
from .requests import (
    COMPUTE_OPERATIONS,
    ArtifactRef,
    ComputeOperation,
    RequestIdentity,
    identifier,
    scalars,
)


@dataclass(frozen=True)
class Accumulation(Record):
    generation: int = 0
    open: bool = False
    contributions: Sequence[str] = ()
    participating_contributions: int = 0

    def __post_init__(self) -> None:
        integer(self.generation, "generation")
        if type(self.open) is not bool:
            raise TypeError("open must be boolean")
        contributions = names(self.contributions, "contributions")
        for value in contributions:
            identifier(value, "contribution request ID")
        object.__setattr__(self, "contributions", contributions)
        integer(self.participating_contributions, "participating_contributions")
        if self.participating_contributions > len(contributions):
            raise ValueError(
                "participating contributions exceed completed contributions"
            )
        if (self.open or contributions) and self.generation == 0:
            raise ValueError("an accumulation requires a positive generation")


@dataclass(frozen=True)
class Acceptance(Record):
    identity: RequestIdentity
    journal_position: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "identity", record(RequestIdentity, self.identity, "identity")
        )
        integer(self.journal_position, "durable journal position", minimum=1)


@dataclass(frozen=True)
class Failure(Record):
    code: str
    message: str
    execution_effect: Literal["not_started", "completed", "unknown"]
    state_valid: bool
    completed_updates: int | None = None
    accumulation: Accumulation | None = None
    failed_dependencies: Sequence[str] = ()

    def __post_init__(self) -> None:
        text(self.code, "error code")
        text(self.message, "error message")
        if self.execution_effect not in ("not_started", "completed", "unknown"):
            raise ValueError("unknown execution effect")
        if type(self.state_valid) is not bool:
            raise TypeError("state_valid must be boolean")
        if self.execution_effect == "unknown" and self.state_valid:
            raise ValueError(
                "unknown execution effects require invalidating live state"
            )
        if self.completed_updates is not None:
            integer(self.completed_updates, "completed_updates")
        if self.execution_effect == "completed" and self.completed_updates is None:
            raise ValueError("known completed execution must record its update count")
        if self.accumulation is not None:
            object.__setattr__(
                self,
                "accumulation",
                record(Accumulation, self.accumulation, "accumulation"),
            )
        dependencies = names(self.failed_dependencies, "failed_dependencies")
        for value in dependencies:
            identifier(value, "failed dependency")
        object.__setattr__(self, "failed_dependencies", dependencies)


@dataclass(frozen=True)
class ArtifactIssue(Record):
    code: str
    message: str
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        text(self.code, "artifact error code")
        text(self.message, "artifact error message")
        if self.artifact_id is not None:
            identifier(self.artifact_id, "artifact ID")


@dataclass(frozen=True)
class ExecutionReceipt(Record):
    """All workers and required terminal transfers finished; publication may remain."""

    identity: RequestIdentity
    operation: ComputeOperation
    outcome: Literal["completed", "skipped"]
    completed_updates: int
    accumulation: Accumulation = field(default_factory=Accumulation)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "identity", record(RequestIdentity, self.identity, "identity")
        )
        if self.operation not in COMPUTE_OPERATIONS or self.outcome not in (
            "completed",
            "skipped",
        ):
            raise ValueError("unknown execution operation or outcome")
        integer(self.completed_updates, "completed_updates")
        object.__setattr__(
            self,
            "accumulation",
            record(Accumulation, self.accumulation, "accumulation"),
        )

    @classmethod
    def from_result(cls, result: OperationResult) -> ExecutionReceipt:
        return cls(
            result.identity,
            result.operation,
            result.outcome,
            result.completed_updates,
            result.accumulation,
        )


@dataclass(frozen=True)
class OperationResult(Record):
    identity: RequestIdentity
    operation: ComputeOperation
    history_id: str
    outcome: Literal["completed", "skipped"]
    completed_updates: int
    accumulation: Accumulation = field(default_factory=Accumulation)
    metrics: Mapping[str, float | int] = field(default_factory=dict)
    outputs: Mapping[str, ArtifactRef] = field(default_factory=dict)
    traces: Sequence[ArtifactRef] = ()
    artifact_issues: Sequence[ArtifactIssue] = ()
    optim: Mapping[str, Scalar] = field(default_factory=dict)
    loss_params: Mapping[str, Scalar] = field(default_factory=dict)
    grad_scale: float | None = None
    loss: float | int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "identity", record(RequestIdentity, self.identity, "identity")
        )
        if self.operation not in COMPUTE_OPERATIONS or self.outcome not in (
            "completed",
            "skipped",
        ):
            raise ValueError("unknown result operation or outcome")
        identifier(self.history_id, "history ID")
        integer(self.completed_updates, "completed_updates")
        if self.loss is not None:
            number(self.loss, "loss")
            if self.operation == "optim_step":
                raise ValueError("optimizer-only results have no forward loss")
        object.__setattr__(self, "optim", scalars(self.optim, "optim"))
        object.__setattr__(
            self, "loss_params", scalars(self.loss_params, "loss_params")
        )
        if self.operation in ("step", "optim_step"):
            if self.grad_scale is None:
                raise ValueError(
                    "optimizer result must record its effective grad_scale"
                )
            number(self.grad_scale, "grad_scale")
        elif self.grad_scale is not None or self.optim:
            raise ValueError("forward/backward results have no optimizer controls")
        if self.operation == "optim_step" and self.loss_params:
            raise ValueError("optim_step results have no objective controls")
        object.__setattr__(
            self,
            "accumulation",
            record(Accumulation, self.accumulation, "accumulation"),
        )
        metrics = dict(self.metrics)
        for name, value in metrics.items():
            text(name, "metric name")
            number(value, "metric value")
        object.__setattr__(self, "metrics", MappingProxyType(metrics))
        object.__setattr__(
            self,
            "outputs",
            MappingProxyType(
                {
                    text(k, "output name"): record(ArtifactRef, v, k)
                    for k, v in self.outputs.items()
                }
            ),
        )
        object.__setattr__(
            self, "traces", tuple(record(ArtifactRef, v, "trace") for v in self.traces)
        )
        object.__setattr__(
            self,
            "artifact_issues",
            tuple(
                record(ArtifactIssue, v, "artifact issue") for v in self.artifact_issues
            ),
        )


@dataclass(frozen=True)
class RequestStatus(Record):
    identity: RequestIdentity
    state: Literal["uploading", "accepted", "running", "completed", "skipped", "failed"]
    acceptance: Acceptance | None = None
    result: OperationResult | None = None
    failure: Failure | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "identity", record(RequestIdentity, self.identity, "identity")
        )
        if self.state not in (
            "uploading",
            "accepted",
            "running",
            "completed",
            "skipped",
            "failed",
        ):
            raise ValueError("unknown request state")
        for name, kind in (
            ("acceptance", Acceptance),
            ("result", OperationResult),
            ("failure", Failure),
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, record(kind, value, name))
        if (
            self.state in ("accepted", "running", "completed", "skipped")
            and self.acceptance is None
        ):
            raise ValueError("accepted work requires its original durable acceptance")
        if self.state == "uploading" and self.acceptance is not None:
            raise ValueError("uploading is not durable acceptance")
        if (self.state == "failed") != (self.failure is not None):
            raise ValueError("only failed requests carry a failure")
        if (
            self.failure is not None
            and self.failure.execution_effect != "not_started"
            and self.acceptance is None
        ):
            raise ValueError("executed work requires its original durable acceptance")
        if (self.state in ("completed", "skipped")) != (self.result is not None):
            raise ValueError(
                "terminal successful requests require their original result"
            )
        if self.result is not None and self.state != self.result.outcome:
            raise ValueError("result outcome differs from request state")
        for child in (self.acceptance, self.result):
            if child is not None and child.identity != self.identity:
                raise ValueError("request record identity mismatch")
