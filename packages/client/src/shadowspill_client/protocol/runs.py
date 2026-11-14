"""Small persisted run descriptions; mutable execution effects live in requests."""

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

from .._records import Record, record, text
from .requests import ArtifactRef, identifier
from .service import ServiceError

type RunState = Literal["preparing", "ready", "failed", "stopped", "closed"]
type CreationStage = Literal[
    "queued", "build", "profile", "communicate", "plan", "initializing"
]
CREATION_STAGES = ("queued", "build", "profile", "communicate", "plan", "initializing")


@dataclass(frozen=True)
class RunInfo(Record):
    """Durable lifecycle observations, not proof that GPU state is still valid.

    Configuration and Plans are immutable artifacts. The operation journal owns
    update counters, accumulated gradients and execution/publication failures.
    The initial history identifies creation; later restore segments belong to
    operation results and do not rewrite creation provenance.
    ``request_sha256`` pins the original creation request for deduplication.
    """

    owner_id: str
    session_id: str
    run_id: str
    server_incarnation: str
    request_sha256: str
    configuration: ArtifactRef
    state: RunState = "preparing"
    stage: CreationStage | None = "queued"
    name: str | None = None
    labels: Mapping[str, str] = field(default_factory=dict)
    plans: ArtifactRef | None = None
    initial_history_id: str | None = None
    error: ServiceError | None = None

    def __post_init__(self) -> None:
        for name in ("owner_id", "session_id", "run_id", "server_incarnation"):
            identifier(getattr(self, name), name)
        if (
            not isinstance(self.request_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.request_sha256) is None
        ):
            raise ValueError("creation request requires a SHA-256 digest")
        object.__setattr__(
            self,
            "configuration",
            record(ArtifactRef, self.configuration, "configuration"),
        )
        if self.plans is not None:
            object.__setattr__(self, "plans", record(ArtifactRef, self.plans, "plans"))
        if self.initial_history_id is not None:
            identifier(self.initial_history_id, "history ID")
        if self.error is not None:
            object.__setattr__(self, "error", record(ServiceError, self.error, "error"))
        if self.state not in ("preparing", "ready", "failed", "stopped", "closed"):
            raise ValueError("unknown run lifecycle state")
        if (self.stage in CREATION_STAGES) != (self.state == "preparing") or (
            self.state != "preparing" and self.stage is not None
        ):
            raise ValueError("only preparing runs have a creation stage")
        if (self.error is not None) != (self.state in ("failed", "stopped")):
            raise ValueError("failed or stopped runs require a lifecycle diagnostic")
        if (self.plans is None) != (self.initial_history_id is None) or (
            self.state in ("ready", "closed") and self.plans is None
        ):
            raise ValueError("initialized runs require Plans and a history identity")
        if self.state in ("preparing", "failed") and self.plans is not None:
            raise ValueError("unfinished creation has no initialized run")
        if self.name is not None and len(text(self.name, "run name")) > 256:
            raise ValueError("run name exceeds its bound")
        if not isinstance(self.labels, Mapping) or len(self.labels) > 16:
            raise ValueError("run labels must be a mapping of at most 16 items")
        for key, value in self.labels.items():
            if (
                len(text(key, "label key")) > 64
                or len(text(value, "label value")) > 256
            ):
                raise ValueError("run label exceeds its bound")
        object.__setattr__(self, "labels", MappingProxyType(dict(self.labels)))
