"""Lightweight discovery of immutable Program-corpus cases."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from benchmarking.program_collection.corpus import ProgramCaseIdentity


@dataclass(frozen=True, slots=True)
class CorpusProgramCase:
    """One saved Program without eagerly parsing its large artifact."""

    directory: Path
    identity: ProgramCaseIdentity
    program_digest: str
    artifact_digest: str

    @property
    def case_id(self) -> str:
        identity = self.identity
        return (
            f"{identity.provider}-{identity.family}__"
            f"{identity.data_geometry.identity_fragment}__"
            f"program-{self.program_digest[:12]}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "directory": str(self.directory),
            "identity": self.identity.to_dict(),
            "data_geometry": self.identity.data_geometry.to_dict(),
            "program_digest": self.program_digest,
            "artifact_digest": self.artifact_digest,
        }


def discover_program_cases(
    corpus_root: Path,
    *,
    expected_count: int | None = None,
) -> tuple[CorpusProgramCase, ...]:
    """Discover cases through small manifests; workers validate full Programs."""

    root = corpus_root.expanduser().resolve()
    cases = tuple(
        sorted(
            (
                _case_from_manifest(path)
                for path in root.glob("cases/*/*/*/manifest.json")
            ),
            key=lambda item: (
                item.identity.tokens_per_microbatch,
                item.identity.sequence_length,
                item.identity.accumulation_rounds,
                item.identity.provider,
                item.identity.family,
                item.program_digest,
            ),
        )
    )
    if not cases:
        raise ValueError(f"no saved Program cases found under {root}")
    if expected_count is not None and len(cases) != expected_count:
        raise ValueError(
            "Program corpus count differs from frontier config: "
            f"expected={expected_count}, observed={len(cases)}"
        )
    case_ids = tuple(item.case_id for item in cases)
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Program corpus contains duplicate case identities")
    return cases


def corpus_manifest_digest(cases: tuple[CorpusProgramCase, ...]) -> str:
    # A corpus is identified by its immutable contents, not by where those
    # contents happen to be mounted.  Keep absolute directories in evidence
    # manifests for navigation, but exclude them from the reusable identity.
    payload = [
        {
            "case_id": item.case_id,
            "identity": item.identity.to_dict(),
            "program_digest": item.program_digest,
            "artifact_digest": item.artifact_digest,
        }
        for item in cases
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _case_from_manifest(path: Path) -> CorpusProgramCase:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read Program manifest {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Program manifest {path} is not an object")
    identity = ProgramCaseIdentity.from_value(value.get("identity"))
    artifact = value.get("step_program")
    if not isinstance(artifact, dict):
        raise ValueError(f"Program manifest {path} has no step_program object")
    program_digest = artifact.get("program_digest")
    artifact_digest = artifact.get("artifact_sha256")
    if not isinstance(program_digest, str) or len(program_digest) != 64:
        raise ValueError(f"Program manifest {path} has an invalid Program digest")
    if not isinstance(artifact_digest, str) or len(artifact_digest) != 64:
        raise ValueError(f"Program manifest {path} has an invalid artifact digest")
    return CorpusProgramCase(
        directory=path.parent.resolve(),
        identity=identity,
        program_digest=program_digest,
        artifact_digest=artifact_digest,
    )


__all__ = [
    "CorpusProgramCase",
    "corpus_manifest_digest",
    "discover_program_cases",
]
