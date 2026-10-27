"""Human-navigable storage for reusable pre-PressureFit Programs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from benchmarking._serialization import (
    atomic_json,
    atomic_text,
    integer,
    mapping,
    string,
)
from benchmarking.data_geometry import DataGeometry

_CORPUS_SCHEMA = "shadowspill.planning_corpus/v2"
CASE_SCHEMA = "shadowspill.planning_corpus.case/v3"
PROGRAM_FILENAME = "profiled_program.ssp"


@dataclass(frozen=True, slots=True)
class ProgramCaseIdentity:
    """Human-readable model and data geometry for one reusable ShadowSpillProgram."""

    family: str
    provider: str
    tokens_per_microbatch: int
    sequence_length: int
    accumulation_rounds: int

    def __post_init__(self) -> None:
        if not self.family or not self.provider:
            raise ValueError("corpus family and provider must be non-empty")
        for name, value in (
            ("tokens_per_microbatch", self.tokens_per_microbatch),
            ("sequence_length", self.sequence_length),
            ("accumulation_rounds", self.accumulation_rounds),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.tokens_per_microbatch % self.sequence_length:
            raise ValueError("sequence length must divide tokens per microbatch")

    @property
    def data_geometry(self) -> DataGeometry:
        return DataGeometry(
            sequence_length=self.sequence_length,
            tokens_per_microbatch=self.tokens_per_microbatch,
            accumulation_rounds=self.accumulation_rounds,
        )

    @property
    def case_name(self) -> str:
        return f"{_safe(self.provider)}-{_safe(self.family)}"

    @property
    def geometry_name(self) -> str:
        return self.data_geometry.identity_fragment.replace("__", "_")

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "provider": self.provider,
            "tokens_per_microbatch": self.tokens_per_microbatch,
            "sequence_length": self.sequence_length,
            "sequences_per_microbatch": (
                self.tokens_per_microbatch // self.sequence_length
            ),
            "accumulation_rounds": self.accumulation_rounds,
            "tokens_per_step": self.tokens_per_microbatch * self.accumulation_rounds,
        }

    @classmethod
    def from_value(cls, value: object) -> ProgramCaseIdentity:
        data = mapping(value, "case.identity")
        raw_rounds = data.get("accumulation_rounds")
        return cls(
            family=string(data.get("family"), "case.identity.family"),
            provider=string(data.get("provider"), "case.identity.provider"),
            tokens_per_microbatch=integer(
                data.get("tokens_per_microbatch"),
                "case.identity.tokens_per_microbatch",
            ),
            sequence_length=integer(
                data.get("sequence_length"), "case.identity.sequence_length"
            ),
            accumulation_rounds=integer(
                raw_rounds, "case.identity.accumulation_rounds"
            ),
        )


@dataclass(frozen=True, slots=True)
class SavedProgramCase:
    """Validated location and identity of one saved pre-PressureFit artifact."""

    directory: Path
    identity: ProgramCaseIdentity
    program_digest: str
    artifact_digest: str

    @property
    def program_path(self) -> Path:
        return self.directory / PROGRAM_FILENAME


def initialize_corpus(root: Path) -> None:
    """Create the stable input-dataset layout and its concise guide."""

    root.mkdir(parents=True, exist_ok=True)
    (root / "cases").mkdir(exist_ok=True)
    layout = root / "layout.json"
    if not layout.exists():
        atomic_json(
            layout,
            {
                "schema": _CORPUS_SCHEMA,
                "directories": {
                    "cases": "model/data-geometry pre-PressureFit Programs"
                },
            },
        )
    guide = root / "README.md"
    if not guide.exists():
        atomic_text(guide, _README)


def _safe(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    if not normalized:
        raise ValueError("corpus path label has no safe characters")
    return normalized


_README = """# Planning input dataset

Each case stores an immutable manifest and a self-contained phase archive
(`profiled_program.ssp`). Cases are grouped by provider/model,
data geometry and content digest. Checksums cover each saved artifact.

Collection builds and measures tasks with the ordinary PyTorch allocator.
Execution budgets, transfer assumptions and planning results belong in the
separate planning-evaluation output. Compiler caches remain external.
"""


__all__ = [
    "ProgramCaseIdentity",
    "SavedProgramCase",
    "initialize_corpus",
]
