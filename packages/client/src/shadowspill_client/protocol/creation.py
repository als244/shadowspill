"""Immutable creation choices, separate from a computation's resolved settings."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from .._records import Record, record
from .requests import ArtifactRef
from .runs import organization
from .training import ResolvedTraining


@dataclass(frozen=True)
class TrainingCreation(Record):
    config: ResolvedTraining
    plans: ArtifactRef | None = None
    checkpoint: ArtifactRef | None = None
    wait_for_resources: bool = False
    name: str | None = None
    labels: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "config", record(ResolvedTraining, self.config, "config")
        )
        for name in ("plans", "checkpoint"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, record(ArtifactRef, value, name))
        if type(self.wait_for_resources) is not bool:
            raise TypeError("wait_for_resources must be boolean")
        object.__setattr__(self, "labels", organization(self.name, self.labels))
