"""Model and computation references; construction happens on the server."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from .._records import Record, Value, integer, mapping, names, number, record, text


@dataclass(frozen=True)
class Weights(Record):
    kind: Literal["random", "pretrained", "artifact"]
    seed: int | None = None
    reference: str | None = None
    revision: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "random":
            integer(self.seed, "seed")
            if self.reference is not None or self.revision is not None:
                raise ValueError("random initialization has no reference or revision")
        elif self.kind in ("pretrained", "artifact"):
            text(self.reference, "reference")
            if self.seed is not None:
                raise ValueError("referenced weights cannot also select a random seed")
            if self.revision is not None:
                if self.kind != "pretrained":
                    raise ValueError("only pretrained references accept a revision")
                text(self.revision, "revision")
        else:
            raise ValueError(f"unknown weights kind: {self.kind}")

    @classmethod
    def random(cls, *, seed: int) -> Weights:
        return cls("random", seed=seed)

    @classmethod
    def pretrained(cls, reference: str, *, revision: str | None = None) -> Weights:
        return cls("pretrained", reference=reference, revision=revision)

    @classmethod
    def artifact(cls, artifact_id: str) -> Weights:
        return cls("artifact", reference=artifact_id)


@dataclass(frozen=True)
class LoRA(Record):
    rank: int
    alpha: float | None = None
    dropout: float = 0.0
    targets: Sequence[str] | None = None

    def __post_init__(self) -> None:
        integer(self.rank, "rank", minimum=1)
        alpha = self.rank if self.alpha is None else self.alpha
        if number(alpha, "alpha") <= 0:
            raise ValueError("alpha must be positive")
        object.__setattr__(self, "alpha", float(alpha))
        if not 0 <= number(self.dropout, "dropout") < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.targets is not None:
            targets = names(self.targets, "targets")
            if not targets:
                raise ValueError("explicit LoRA targets must not be empty")
            object.__setattr__(self, "targets", targets)


@dataclass(frozen=True)
class Model(Record):
    definition: str | None = None
    config: Mapping[str, Value] = field(default_factory=dict)
    initialize: Weights | None = None
    lora: LoRA | None = None
    implementation: str | None = None
    precision: Mapping[str, Value] = field(default_factory=dict)
    trainable: Sequence[str] | None = None

    def __post_init__(self) -> None:
        if self.definition is not None:
            text(self.definition, "definition")
        if self.initialize is not None:
            object.__setattr__(
                self, "initialize", record(Weights, self.initialize, "initialize")
            )
        if self.definition is None and (
            self.initialize is None or self.initialize.kind != "pretrained"
        ):
            raise ValueError("select an installed definition or a pretrained model")
        if self.lora is not None:
            object.__setattr__(self, "lora", record(LoRA, self.lora, "lora"))
        if self.implementation is not None:
            text(self.implementation, "implementation")
        object.__setattr__(self, "config", mapping(self.config, "config"))
        object.__setattr__(self, "precision", mapping(self.precision, "precision"))
        if self.trainable is not None:
            object.__setattr__(self, "trainable", names(self.trainable, "trainable"))

    @classmethod
    def from_pretrained(
        cls,
        reference: str,
        *,
        revision: str | None = None,
        lora: LoRA | None = None,
        implementation: str | None = None,
        config: Mapping[str, Value] | None = None,
        precision: Mapping[str, Value] | None = None,
    ) -> Model:
        return cls(
            initialize=Weights.pretrained(reference, revision=revision),
            lora=lora,
            implementation=implementation,
            config=config or {},
            precision=precision or {},
        )


@dataclass(frozen=True)
class Objective(Record):
    definition: str
    config: Mapping[str, Value] = field(default_factory=dict)
    defaults: Mapping[str, Value] = field(default_factory=dict)

    def __post_init__(self) -> None:
        text(self.definition, "definition")
        object.__setattr__(self, "config", mapping(self.config, "config"))
        object.__setattr__(self, "defaults", mapping(self.defaults, "defaults"))


@dataclass(frozen=True)
class Optimizer(Record):
    definition: str
    defaults: Mapping[str, Value] = field(default_factory=dict)
    config: Mapping[str, Value] = field(default_factory=dict)
    groups: Mapping[str, Value] = field(default_factory=dict)
    precision: Mapping[str, Value] = field(default_factory=dict)

    def __post_init__(self) -> None:
        text(self.definition, "definition")
        for name in ("defaults", "config", "groups", "precision"):
            object.__setattr__(self, name, mapping(getattr(self, name), name))
