"""Human-navigable storage for reusable planning Programs and selections."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shadowspill.pytorch import AnnotatedProgramPlan, StepProgram

_CORPUS_SCHEMA = "shadowspill.planning_corpus/v1"
_CASE_SCHEMA = "shadowspill.planning_corpus.case/v1"
_SELECTION_SCHEMA = "shadowspill.planning_corpus.selection/v1"


@dataclass(frozen=True, slots=True)
class ProgramCaseIdentity:
    """Human-readable model and data geometry for one reusable Program."""

    family: str
    provider: str
    tokens_per_microbatch: int
    sequence_length: int
    accumulation_steps: int

    def __post_init__(self) -> None:
        if not self.family or not self.provider:
            raise ValueError("corpus family and provider must be non-empty")
        for name, value in (
            ("tokens_per_microbatch", self.tokens_per_microbatch),
            ("sequence_length", self.sequence_length),
            ("accumulation_steps", self.accumulation_steps),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.tokens_per_microbatch % self.sequence_length:
            raise ValueError("sequence length must divide tokens per microbatch")

    @property
    def case_name(self) -> str:
        return f"{_safe(self.provider)}-{_safe(self.family)}"

    @property
    def geometry_name(self) -> str:
        return (
            f"tokens-{self.tokens_per_microbatch}_"
            f"sequence-{self.sequence_length}_"
            f"accumulation-{self.accumulation_steps}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "provider": self.provider,
            "tokens_per_microbatch": self.tokens_per_microbatch,
            "sequence_length": self.sequence_length,
            "sequences_per_microbatch": (
                self.tokens_per_microbatch // self.sequence_length
            ),
            "accumulation_steps": self.accumulation_steps,
            "tokens_per_step": (
                self.tokens_per_microbatch * self.accumulation_steps
            ),
        }

    @classmethod
    def from_value(cls, value: object) -> ProgramCaseIdentity:
        data = _mapping(value, "case.identity")
        return cls(
            family=_string(data.get("family"), "case.identity.family"),
            provider=_string(data.get("provider"), "case.identity.provider"),
            tokens_per_microbatch=_integer(
                data.get("tokens_per_microbatch"),
                "case.identity.tokens_per_microbatch",
            ),
            sequence_length=_integer(
                data.get("sequence_length"), "case.identity.sequence_length"
            ),
            accumulation_steps=_integer(
                data.get("accumulation_steps"),
                "case.identity.accumulation_steps",
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
    """Create the stable corpus layout and its concise human guide."""

    root.mkdir(parents=True, exist_ok=True)
    (root / "cases").mkdir(exist_ok=True)
    layout = root / "layout.json"
    if not layout.exists():
        _atomic_json(
            layout,
            {
                "schema": _CORPUS_SCHEMA,
                "directories": {
                    "cases": (
                        "model/data-geometry Programs and their independently "
                        "selected annotated plans"
                    )
                },
            },
        )
    guide = root / "README.md"
    if not guide.exists():
        _atomic_text(guide, _README)


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
    artifact_digest = _text_digest(payload)
    directory = (
        root
        / "cases"
        / identity.case_name
        / identity.geometry_name
        / program.digest
    )
    directory.mkdir(parents=True, exist_ok=True)
    program_path = directory / "step_program.json"
    manifest_path = directory / "manifest.json"
    manifest = {
        "schema": _CASE_SCHEMA,
        "identity": identity.to_dict(),
        "metadata": _json_mapping(metadata or {}, "metadata"),
        "step_program": {
            "path": "step_program.json",
            "program_digest": program.digest,
            "artifact_sha256": artifact_digest,
        },
        "evaluations_directory": "evaluations",
    }
    if not _existing_artifact_is_identical(
        program_path,
        manifest_path,
        payload=payload,
        manifest=manifest,
    ):
        _atomic_text(program_path, _pretty_json(payload))
        _atomic_json(manifest_path, manifest)
    (directory / "evaluations").mkdir(exist_ok=True)
    return SavedProgramCase(
        directory.resolve(), identity, program.digest, artifact_digest
    )


def load_step_program(path: Path) -> tuple[SavedProgramCase, StepProgram]:
    """Load and integrity-check a case directory or its Program JSON file."""

    directory = path if path.is_dir() else path.parent
    manifest = _read_mapping(directory / "manifest.json", "case manifest")
    if manifest.get("schema") != _CASE_SCHEMA:
        raise ValueError("case manifest has an unsupported schema")
    identity = ProgramCaseIdentity.from_value(manifest.get("identity"))
    artifact = _mapping(manifest.get("step_program"), "case.step_program")
    program_path = directory / _string(
        artifact.get("path"), "case.step_program.path"
    )
    payload = program_path.read_text()
    expected_artifact = _string(
        artifact.get("artifact_sha256"), "case.step_program.artifact_sha256"
    )
    if _text_digest(_canonicalize_json(payload)) != expected_artifact:
        raise ValueError("saved StepProgram artifact digest does not match")
    program = StepProgram.from_json(payload)
    expected_program = _string(
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


def save_annotated_plan(
    case: SavedProgramCase,
    plan: AnnotatedProgramPlan,
    *,
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """Save one selected plan under independent budget and bandwidth axes."""

    saved_case, step_program = load_step_program(case.directory)
    if saved_case != case:
        raise ValueError("saved Program case identity changed on disk")
    program_digests = {step_program.recurrent.program.digest}
    if step_program.initial is not None:
        program_digests.add(step_program.initial.program.digest)
    if plan.program.program.digest not in program_digests:
        raise ValueError("annotated plan does not belong to the saved StepProgram")
    budgets = plan.memory_budgets
    bandwidths = plan.transfer_bandwidths
    directory = (
        case.directory
        / "evaluations"
        / (
            f"execution-{budgets.execution_bytes}_"
            f"spill-{budgets.spill_bytes}"
        )
        / (
            f"fetch-{bandwidths.fetch_bytes_per_second}_"
            f"evict-{bandwidths.evict_bytes_per_second}"
        )
        / plan.digest
    )
    directory.mkdir(parents=True, exist_ok=True)
    payload = plan.to_json()
    artifact_digest = _text_digest(payload)
    plan_path = directory / "annotated_program_plan.json"
    manifest_path = directory / "manifest.json"
    manifest = {
        "schema": _SELECTION_SCHEMA,
        "source_step_program_digest": case.program_digest,
        "source_pressurefit_program_digest": plan.program.digest,
        "memory_budgets": plan.memory_budgets.to_dict(),
        "transfer_bandwidths": plan.transfer_bandwidths.to_dict(),
        "annotated_program_plan": {
            "path": "annotated_program_plan.json",
            "plan_digest": plan.digest,
            "artifact_sha256": artifact_digest,
        },
        "metadata": _json_mapping(metadata or {}, "metadata"),
    }
    if not _existing_artifact_is_identical(
        plan_path,
        manifest_path,
        payload=payload,
        manifest=manifest,
    ):
        _atomic_text(plan_path, _pretty_json(payload))
        _atomic_json(manifest_path, manifest)
    return directory.resolve()


def load_annotated_plan(path: Path) -> AnnotatedProgramPlan:
    """Load and integrity-check an annotated-plan directory or JSON file."""

    directory = path if path.is_dir() else path.parent
    manifest = _read_mapping(directory / "manifest.json", "selection manifest")
    if manifest.get("schema") != _SELECTION_SCHEMA:
        raise ValueError("selection manifest has an unsupported schema")
    artifact = _mapping(
        manifest.get("annotated_program_plan"),
        "selection.annotated_program_plan",
    )
    plan_path = directory / _string(
        artifact.get("path"), "selection.annotated_program_plan.path"
    )
    payload = plan_path.read_text()
    expected_artifact = _string(
        artifact.get("artifact_sha256"),
        "selection.annotated_program_plan.artifact_sha256",
    )
    if _text_digest(_canonicalize_json(payload)) != expected_artifact:
        raise ValueError("saved annotated-plan artifact digest does not match")
    plan = AnnotatedProgramPlan.from_json(payload)
    expected_plan = _string(
        artifact.get("plan_digest"), "selection.annotated_program_plan.plan_digest"
    )
    if plan.digest != expected_plan:
        raise ValueError("saved annotated-plan content digest does not match")
    if manifest.get("memory_budgets") != plan.memory_budgets.to_dict():
        raise ValueError("selection manifest memory budgets do not match the plan")
    if manifest.get("transfer_bandwidths") != plan.transfer_bandwidths.to_dict():
        raise ValueError("selection manifest transfer bandwidths do not match the plan")
    return plan


def _safe(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    if not normalized:
        raise ValueError("corpus path label has no safe characters")
    return normalized


def _canonicalize_json(payload: str) -> str:
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError("saved JSON artifact is invalid") from error
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _pretty_json(payload: str) -> str:
    value = json.loads(payload)
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _text_digest(payload: str) -> str:
    return hashlib.sha256(_canonicalize_json(payload).encode()).hexdigest()


def _json_mapping(value: Mapping[str, object], path: str) -> dict[str, object]:
    try:
        encoded = json.dumps(dict(value), sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path} must contain only JSON values") from error
    decoded = json.loads(encoded)
    return _mapping(decoded, path)


def _read_mapping(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} at {path} cannot be read") from error
    return _mapping(value, name)


def _existing_artifact_is_identical(
    artifact_path: Path,
    manifest_path: Path,
    *,
    payload: str,
    manifest: Mapping[str, object],
) -> bool:
    """Return true for an identical retry and reject immutable collisions."""

    artifact_exists = artifact_path.exists()
    manifest_exists = manifest_path.exists()
    if not artifact_exists and not manifest_exists:
        return False
    if artifact_exists != manifest_exists:
        raise FileExistsError(
            f"incomplete immutable corpus artifact at {artifact_path.parent}"
        )
    existing_payload = _canonicalize_json(artifact_path.read_text())
    existing_manifest = _read_mapping(manifest_path, "existing manifest")
    if existing_payload != _canonicalize_json(payload) or (
        _canonical_json_value(existing_manifest) != _canonical_json_value(manifest)
    ):
        raise FileExistsError(
            "immutable corpus identity already contains different evidence: "
            f"{artifact_path.parent}"
        )
    return True


def _mapping(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{path}: expected an object")
    return value


def _canonical_json_value(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _string(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{path}: expected a string")
    return value


def _integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path}: expected an integer")
    return value


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


_README = """# ShadowSpill planning corpus

Each case stores one self-contained `step_program.json` produced before
PressureFit. Its directory is organized by provider/model, data geometry, and
content digest.

`evaluations/` contains independently selected annotated plans. Memory budgets
and transfer bandwidths are separate path and manifest dimensions; they are
also first-class fields in every `AnnotatedProgramPlan`. Benchmark labels and
comparison results belong to the corpus manifests, not to the general plan
object.

Every JSON file has a SHA-256 entry in its adjacent manifest and is validated
again by the corresponding `from_json()` constructor when loaded.
"""


__all__ = [
    "ProgramCaseIdentity",
    "SavedProgramCase",
    "initialize_corpus",
    "load_annotated_plan",
    "load_step_program",
    "save_annotated_plan",
    "save_step_program",
]
