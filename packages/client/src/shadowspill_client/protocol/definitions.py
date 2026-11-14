"""Inspectable extension identities, output ordering and participation rules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

from .._records import Record, names, record, text
from .inputs import DTYPES, shape


@dataclass(frozen=True)
class DefinitionSource(Record):
    distribution: str
    version: str
    digest: str

    def __post_init__(self) -> None:
        text(self.distribution, "distribution")
        text(self.version, "version")
        if len(self.digest) != 64 or any(
            c not in "0123456789abcdef" for c in self.digest
        ):
            raise ValueError("definition source requires a lowercase SHA-256 digest")


@dataclass(frozen=True)
class TensorOutput(Record):
    dtype: str
    shape: Sequence[int | str]
    ordering: Literal["datum", "aligned", "global"]
    alignment: str | None = None
    source: Sequence[str] | None = None
    reduction: Literal["sum", "min", "max", "identical"] | None = None

    def __post_init__(self) -> None:
        if self.dtype not in DTYPES:
            raise ValueError("unsupported output dtype")
        object.__setattr__(self, "shape", shape(self.shape))
        if self.ordering not in ("datum", "aligned", "global"):
            raise ValueError("output ordering must be datum, aligned or global")
        if self.ordering == "aligned":
            text(self.alignment, "output alignment")
            if self.alignment not in self.shape:
                raise ValueError("aligned output must name a declared dimension")
        elif self.alignment is not None:
            raise ValueError("only aligned outputs declare an alignment")
        if self.source is not None:
            if (
                not isinstance(self.source, (list, tuple))
                or len(self.source) < 2
                or self.source[0] not in ("inputs", "loss_inputs")
                or not any(isinstance(d, str) for d in self.shape)
                or self.ordering == "global"
            ):
                raise ValueError("symbolic output source must name a model/loss field")
            object.__setattr__(
                self, "source", tuple(text(p, "output source") for p in self.source)
            )
        if self.ordering == "global":
            if self.reduction not in ("sum", "min", "max", "identical"):
                raise ValueError("declare a reduction for every global tensor output")
            if any(type(d) is not int for d in self.shape):
                raise ValueError("global tensor shapes must have fixed dimensions")
        elif self.reduction is not None:
            raise ValueError("only global tensor outputs declare a reduction")


@dataclass(frozen=True)
class OutputSet(Record):
    """One prepared output combination and its required fields/controls.

    Paths are explicit component tuples. Optional, known input fields may still
    be submitted, but a selected set cannot silently depend on omitted fields.
    Scalar metrics declare reductions; tensor outputs retain their inverse map.
    """

    model_inputs: Sequence[Sequence[str]] = ()
    loss_inputs: Sequence[Sequence[str]] = ()
    controls: Sequence[str] = ()
    tensors: Mapping[str, TensorOutput] = field(default_factory=dict)
    metrics: Mapping[str, Literal["sum", "min", "max", "identical"]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        for name in ("model_inputs", "loss_inputs"):
            paths = []
            for path in getattr(self, name):
                if not isinstance(path, (list, tuple)) or not path:
                    raise ValueError("output input paths must be nonempty sequences")
                paths.append(tuple(text(p, name) for p in path))
            if len(set(paths)) != len(paths):
                raise ValueError(f"duplicate {name} path")
            object.__setattr__(self, name, tuple(paths))
        object.__setattr__(self, "controls", names(self.controls, "output controls"))
        tensors = {
            text(k, "output name"): record(TensorOutput, v, k)
            for k, v in self.tensors.items()
        }
        metrics = {text(k, "metric name"): v for k, v in self.metrics.items()}
        if set(tensors) & set(metrics):
            raise ValueError("tensor outputs and scalar metrics require distinct names")
        if any(v not in ("sum", "min", "max", "identical") for v in metrics.values()):
            raise ValueError(
                "unsupported metric reduction; report means from declared sums"
            )
        object.__setattr__(self, "tensors", MappingProxyType(tensors))
        object.__setattr__(self, "metrics", MappingProxyType(metrics))


@dataclass(frozen=True)
class Participation(Record):
    """Validity, independent of numeric loss, weight, advantage or gradient scale.

    With a path, true entries in the declared boolean loss field contribute.
    Without a path, each real Datum contributes. Padding never participates.
    More elaborate custom participation is an installed CPU preparation hook.
    """

    path: Sequence[str] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.path, (list, tuple)):
            raise TypeError("participation path must be a sequence")
        object.__setattr__(
            self, "path", tuple(text(p, "participation") for p in self.path)
        )
