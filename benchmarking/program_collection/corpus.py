"""Human-navigable storage for reusable pre-PressureFit Programs."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from benchmarking._serialization import (
    atomic_json,
    atomic_text,
    commit_immutable_directory,
    existing_artifact_is_identical,
    integer,
    json_mapping,
    mapping,
    pretty_json,
    read_mapping,
    string,
    text_digest,
)
from benchmarking.data_geometry import DataGeometry
from shadowspill.pytorch import StepProgram

_CORPUS_SCHEMA = "shadowspill.planning_corpus/v1"
_CASE_SCHEMA = "shadowspill.planning_corpus.case/v1"


@dataclass(frozen=True, slots=True)
class ProgramCaseIdentity:
    """Human-readable model and data geometry for one reusable Program."""

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
            # Schema v1 retains this key so the collected dataset remains
            # immutable. New interfaces and reports say "accumulation rounds".
            "accumulation_steps": self.accumulation_rounds,
            "tokens_per_step": self.tokens_per_microbatch * self.accumulation_rounds,
        }

    @classmethod
    def from_value(cls, value: object) -> ProgramCaseIdentity:
        data = mapping(value, "case.identity")
        raw_rounds = data.get("accumulation_rounds", data.get("accumulation_steps"))
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
        return self.directory / "step_program.json"


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


def save_step_program(
    root: Path,
    *,
    identity: ProgramCaseIdentity,
    program: StepProgram,
    metadata: Mapping[str, object] | None = None,
) -> SavedProgramCase:
    """Atomically save one self-contained pre-PressureFit Program case."""

    initialize_corpus(root)
    payload = program.to_json()
    artifact_digest = text_digest(payload)
    directory = (
        root / "cases" / identity.case_name / identity.geometry_name / program.digest
    )
    manifest = {
        "schema": _CASE_SCHEMA,
        "identity": identity.to_dict(),
        "metadata": json_mapping(metadata or {}, "metadata"),
        "step_program": {
            "path": "step_program.json",
            "program_digest": program.digest,
            "artifact_sha256": artifact_digest,
        },
        # Retained in schema v1 for byte-compatible corpus retries. Planner
        # evaluations now write only beneath benchmarking/planning_eval/results.
        "evaluations_directory": "evaluations",
    }
    program_path = directory / "step_program.json"
    manifest_path = directory / "manifest.json"
    if not existing_artifact_is_identical(
        program_path,
        manifest_path,
        payload=payload,
        manifest=manifest,
    ):
        commit_immutable_directory(
            directory,
            artifact_name="step_program.json",
            payload=pretty_json(payload),
            manifest=manifest,
            child_directories=("evaluations",),
        )
    return SavedProgramCase(
        directory.resolve(), identity, program.digest, artifact_digest
    )


def load_step_program(path: Path) -> tuple[SavedProgramCase, StepProgram]:
    """Load and integrity-check a case directory or its Program JSON file."""

    directory = path if path.is_dir() else path.parent
    manifest = read_mapping(directory / "manifest.json", "case manifest")
    if manifest.get("schema") != _CASE_SCHEMA:
        raise ValueError("case manifest has an unsupported schema")
    identity = ProgramCaseIdentity.from_value(manifest.get("identity"))
    artifact = mapping(manifest.get("step_program"), "case.step_program")
    program_path = directory / string(artifact.get("path"), "case.step_program.path")
    payload = program_path.read_text()
    expected_artifact = string(
        artifact.get("artifact_sha256"), "case.step_program.artifact_sha256"
    )
    if text_digest(payload) != expected_artifact:
        raise ValueError("saved StepProgram artifact digest does not match")
    program = StepProgram.from_json(payload)
    expected_program = string(
        artifact.get("program_digest"), "case.step_program.program_digest"
    )
    if program.digest != expected_program:
        raise ValueError("saved StepProgram content digest does not match")
    return (
        SavedProgramCase(
            directory.resolve(), identity, program.digest, expected_artifact
        ),
        program,
    )


def _safe(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    if not normalized:
        raise ValueError("corpus path label has no safe characters")
    return normalized


_README = """# ShadowSpill input Program dataset

Each case stores one self-contained `step_program.json` produced before
PressureFit. Directories are organized by provider/model, data geometry, and
content digest. Planning-evaluation results live outside this immutable input
dataset under `benchmarking/planning_eval/results/`.

Every Program has a SHA-256 entry in its adjacent manifest and is validated by
`StepProgram.from_json()` when loaded.
"""


__all__ = [
    "ProgramCaseIdentity",
    "SavedProgramCase",
    "initialize_corpus",
    "load_step_program",
    "save_step_program",
]
