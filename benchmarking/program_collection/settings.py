"""Strict configuration schema for reusable ShadowSpillProgram collection."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass

from workloads.models import catalog
from workloads.providers import ModelImplementation

_SCHEMA = "shadowspill.program_corpus_collection/v2"


@dataclass(frozen=True, slots=True)
class GeometryAxes:
    """Ordered geometry axes expanded with divisibility validation."""

    tokens_per_microbatch: tuple[int, ...]
    sequence_lengths: tuple[int, ...]
    accumulation_rounds: tuple[int, ...]

    def __post_init__(self) -> None:
        for name, values in (
            ("tokens_per_microbatch", self.tokens_per_microbatch),
            ("sequence_lengths", self.sequence_lengths),
            ("accumulation_rounds", self.accumulation_rounds),
        ):
            if not values:
                raise ValueError(f"geometry.{name} must not be empty")
            if any(value <= 0 for value in values):
                raise ValueError(f"geometry.{name} values must be positive")
            if len(values) != len(set(values)):
                raise ValueError(f"geometry.{name} contains duplicates")

    def to_dict(self) -> dict[str, object]:
        return {
            "tokens_per_microbatch": list(self.tokens_per_microbatch),
            "sequence_lengths": list(self.sequence_lengths),
            "accumulation_rounds": list(self.accumulation_rounds),
        }


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One registered catalog model, its implementation and optional axes."""

    name: str
    model: str
    implementation: ModelImplementation
    geometry: GeometryAxes | None = None
    head_scratch_bytes: int | None = None

    def __post_init__(self) -> None:
        if not self.name or re.search(r"[^A-Za-z0-9._-]", self.name):
            raise ValueError(
                "model name must contain only letters, digits, '.', '_', or '-'"
            )
        entry = catalog.entry(self.model)
        if self.implementation not in entry.implementations:
            raise ValueError(
                f"{self.model} has no {self.implementation!r} implementation"
            )
        if self.head_scratch_bytes is not None and self.head_scratch_bytes <= 0:
            raise ValueError("model head_scratch_bytes must be positive")

    @property
    def family(self) -> str:
        """The family a collected program's identity records."""

        return catalog.entry(self.model).family

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "model": self.model,
            "implementation": self.implementation,
            "geometry": None if self.geometry is None else self.geometry.to_dict(),
            "head_scratch_bytes": self.head_scratch_bytes,
        }


@dataclass(frozen=True, slots=True)
class ProfilingSpec:
    """Device and native-allocator measurement controls; no execution pools."""

    device: str = "cuda:0"
    warmup_seconds: float = 0.05
    measure_seconds: float = 0.18
    external_headroom_bytes: int = 512 << 20

    def __post_init__(self) -> None:
        if re.fullmatch(r"cuda(?::[0-9]+)?", self.device) is None:
            raise ValueError("profiling.device must name a CUDA device")
        for name in ("warmup_seconds", "measure_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"profiling.{name} must be positive and finite")
        if self.external_headroom_bytes < 0:
            raise ValueError("profiling.external_headroom_bytes must be nonnegative")

    def to_dict(self) -> dict[str, object]:
        return {
            "device": self.device,
            "warmup_seconds": self.warmup_seconds,
            "measure_seconds": self.measure_seconds,
            "external_headroom_bytes": self.external_headroom_bytes,
        }


@dataclass(frozen=True, slots=True)
class CollectionConfig:
    """Complete immutable request for one ShadowSpillProgram corpus collection."""

    name: str
    seed: int
    expected_programs: int | None
    case_timeout_seconds: int
    max_attempts: int
    geometry: GeometryAxes
    models: tuple[ModelSpec, ...]
    profiling: ProfilingSpec

    def __post_init__(self) -> None:
        if not self.name or re.search(r"[^A-Za-z0-9._-]", self.name):
            raise ValueError(
                "collection name must contain only letters, digits, '.', '_', or '-'"
            )
        if not self.models:
            raise ValueError("models must not be empty")
        names = tuple(model.name for model in self.models)
        if len(names) != len(set(names)):
            raise ValueError("model names must be unique")
        if self.expected_programs is not None and self.expected_programs <= 0:
            raise ValueError("expected_programs must be positive")
        if self.case_timeout_seconds <= 0:
            raise ValueError("case_timeout_seconds must be positive")
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")

    @property
    def digest(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": _SCHEMA,
            "name": self.name,
            "seed": self.seed,
            "expected_programs": self.expected_programs,
            "case_timeout_seconds": self.case_timeout_seconds,
            "max_attempts": self.max_attempts,
            "geometry": self.geometry.to_dict(),
            "models": [model.to_dict() for model in self.models],
            "profiling": self.profiling.to_dict(),
        }
