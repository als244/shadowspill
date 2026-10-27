"""Immutable frontier identities and filesystem paths."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from benchmarking._serialization import atomic_json, atomic_text, canonical_json_value
from benchmarking.journals import read_object, utc_now
from shadowspill.schema import artifact_schema

from .config import FrontierConfig
from .matrix import FrontierPointRequest
from .provenance import RepositoryProvenance, environment_provenance
from .source import CorpusProgramCase

_BASELINE_SCHEMA = artifact_schema("search_frontier_baseline")

_README = """# PressureFit frontier baseline

This directory is an immutable planner-implementation identity plus resumable
per-program journals. `config.json` defines the exact budget/bandwidth matrix;
`manifest.json` freezes the program corpus and source revision; `planner.patch`
freezes tracked uncommitted changes. `frontier.csv` and `summary.json` are the
compact comparison index.

Each `cases/<case>/points/<point>/point.json` is the compact outcome for one
program/budget/bandwidth input. Successful points link to a complete canonical
`AnnotatedProgramPlan` under the corresponding baseline case's
`annotated-plans/` tree. The frozen source corpus is never modified by a
frontier run. Each plan artifact contains the selected schedule, simulator
intervals and memory timeline, physical admission certificate, diagnostics,
and source ShadowSpillProgram.
"""


@dataclass(frozen=True, slots=True)
class BaselinePaths:
    """Stable central index for one exact planner implementation."""

    directory: Path
    cases_directory: Path
    config_path: Path
    manifest_path: Path
    summary_path: Path
    csv_path: Path
    jsonl_path: Path
    log_path: Path
    patch_path: Path

    @classmethod
    def open_existing(
        cls,
        directory: Path,
        *,
        config: FrontierConfig,
        corpus_digest: str,
    ) -> BaselinePaths:
        """Open and validate one immutable baseline selected for resume."""

        result = cls._from_directory(directory.expanduser().resolve())
        if not result.directory.is_dir():
            raise ValueError(f"resume baseline does not exist: {result.directory}")
        if canonical_json_value(
            read_object(result.config_path)
        ) != canonical_json_value(config.to_dict()):
            raise ValueError("resume baseline configuration changed")
        manifest = read_object(result.manifest_path)
        if manifest.get("baseline_id") != result.directory.name:
            raise ValueError("resume baseline identity is invalid")
        if manifest.get("config_digest") != config.digest:
            raise ValueError("resume baseline config digest changed")
        corpus = manifest.get("corpus")
        if (
            not isinstance(corpus, dict)
            or corpus.get("manifest_digest") != corpus_digest
        ):
            raise ValueError("resume baseline corpus digest changed")
        return result

    @classmethod
    def _from_directory(cls, directory: Path) -> BaselinePaths:
        return cls(
            directory=directory,
            cases_directory=directory / "cases",
            config_path=directory / "config.json",
            manifest_path=directory / "manifest.json",
            summary_path=directory / "summary.json",
            csv_path=directory / "frontier.csv",
            jsonl_path=directory / "frontier.jsonl",
            log_path=directory / "collection.log",
            patch_path=directory / "planner.patch",
        )

    @classmethod
    def initialize(
        cls,
        output_root: Path,
        *,
        baseline_id: str,
        config: FrontierConfig,
        provenance: RepositoryProvenance,
        corpus_root: Path,
        corpus_digest: str,
        cases: tuple[CorpusProgramCase, ...],
    ) -> BaselinePaths:
        directory = output_root.expanduser().resolve() / baseline_id
        result = cls._from_directory(directory)
        result.cases_directory.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema": _BASELINE_SCHEMA,
            "baseline_id": baseline_id,
            "created_at": utc_now(),
            "config_digest": config.digest,
            "corpus": {
                "root": str(corpus_root.expanduser().resolve()),
                "manifest_digest": corpus_digest,
                "program_count": len(cases),
                "programs": [item.to_dict() for item in cases],
            },
            "repository": provenance.to_dict(),
            "environment": environment_provenance(),
        }
        write_or_match(result.config_path, config.to_dict(), "frontier config")
        if result.manifest_path.exists():
            existing = read_object(result.manifest_path)
            stable_existing = dict(existing)
            stable_manifest = dict(manifest)
            stable_existing.pop("created_at", None)
            stable_manifest.pop("created_at", None)
            stable_existing_environment = stable_existing.get("environment")
            stable_manifest_environment = stable_manifest.get("environment")
            if isinstance(stable_existing_environment, dict):
                stable_existing_environment.pop("pid", None)
            if isinstance(stable_manifest_environment, dict):
                stable_manifest_environment.pop("pid", None)
            if canonical_json_value(stable_existing) != canonical_json_value(
                stable_manifest
            ):
                raise ValueError("baseline manifest identity changed")
        else:
            atomic_json(result.manifest_path, manifest)
        _write_or_match_text(result.patch_path, provenance.diff, "planner patch")
        _write_or_match_text(
            result.directory / "git-status.txt",
            provenance.status,
            "git status",
        )
        _write_or_match_text(result.directory / "README.md", _README, "baseline guide")
        return result

    def case_directory(self, case: CorpusProgramCase) -> Path:
        return self.cases_directory / case.case_id

    def point_directory(
        self,
        case: CorpusProgramCase,
        request: FrontierPointRequest,
    ) -> Path:
        return self.case_directory(case) / "points" / request.point_id


def write_or_match(path: Path, value: Mapping[str, object], name: str) -> None:
    if path.exists():
        if canonical_json_value(read_object(path)) != canonical_json_value(value):
            raise ValueError(f"{name} changed at {path}")
        return
    atomic_json(path, value)


def _write_or_match_text(path: Path, payload: str, name: str) -> None:
    if path.exists():
        if path.read_text() != payload:
            raise ValueError(f"{name} changed at {path}")
        return
    atomic_text(path, payload)
