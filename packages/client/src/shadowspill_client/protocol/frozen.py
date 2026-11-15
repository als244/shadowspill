"""Exact installed initialization provenance for omitted frozen parameters."""

from dataclasses import dataclass
from typing import Literal

from .._records import Record, record
from ..config.model import Model
from .definitions import DefinitionSource
from .requests import ArtifactRef


@dataclass(frozen=True)
class FrozenBase(Record):
    model: Model
    source: DefinitionSource
    artifact: ArtifactRef | None = None
    version: Literal[1] = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", record(Model, self.model, "base model"))
        object.__setattr__(
            self, "source", record(DefinitionSource, self.source, "base source")
        )
        if type(self.version) is not int or self.version != 1:
            raise ValueError("unsupported frozen-base version")
        weights = self.model.initialize
        if self.model.definition is None or weights is None:
            raise ValueError("frozen base requires a resolved installed initializer")
        if weights.kind == "pretrained" and weights.revision is None:
            raise ValueError("frozen base requires a pinned pretrained revision")
        if self.artifact is not None:
            object.__setattr__(
                self, "artifact", record(ArtifactRef, self.artifact, "base artifact")
            )
        if weights.kind == "artifact":
            if self.artifact is None or self.artifact.id != weights.reference:
                raise ValueError("frozen base requires its exact weights artifact")
        elif self.artifact is not None:
            raise ValueError("only artifact initialization has a base artifact")

    @property
    def identity(self) -> str:
        from .codec import digest

        return "base." + digest(self)
