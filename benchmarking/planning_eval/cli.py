"""CLI for reproducible, resumable PressureFit frontier baselines."""

from __future__ import annotations

import argparse
import fnmatch
import json
import shlex
import sys
from pathlib import Path

from .config import load_frontier_config
from .controller import ControllerOptions, run_frontier_collection
from .provenance import (
    RepositoryProvenance,
    capture_repository_provenance,
    resume_provenance_relationship,
)
from .source import (
    CorpusProgramCase,
    corpus_manifest_digest,
    discover_program_cases,
)
from .storage import (
    BaselinePaths,
    append_log,
    atomic_text,
    read_object,
    utc_now,
)


def main() -> int:
    parser = _parser()
    arguments = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[2]
    config = load_frontier_config(arguments.config.expanduser().resolve())
    cases = discover_program_cases(
        arguments.corpus_dir,
        expected_count=config.expected_programs,
    )
    selected = _select_cases(
        cases,
        patterns=tuple(arguments.case),
        start_at=arguments.start_at,
        limit=arguments.limit,
    )
    provenance = capture_repository_provenance(repository_root)
    if provenance.untracked_sources:
        # A tracked diff cannot reproduce these. The baseline records them and
        # says so once; it is not a reason to refuse to run.
        print(
            "note: source files not in git are part of this run and cannot be "
            "reproduced from its recorded diff: "
            + ", ".join(provenance.untracked_sources),
            flush=True,
        )
    corpus_digest = corpus_manifest_digest(cases)
    baseline_id = provenance.baseline_id(config.name, config.digest)
    resume_directory, resume_provenance = _find_resume_baseline(
        arguments.output_dir,
        baseline_id=baseline_id,
        config_digest=config.digest,
        corpus_digest=corpus_digest,
        provenance=provenance,
        enabled=arguments.resume,
        select_revision=arguments.revision,
    )
    if resume_directory is not None:
        baseline_id = resume_directory.name
    matrix = {
        "baseline_id": baseline_id,
        "config_digest": config.digest,
        "corpus_manifest_digest": corpus_digest,
        "total_programs": len(cases),
        "selected_programs": len(selected),
        "points_per_program": config.expected_points_per_program,
        "total_points": len(cases) * config.expected_points_per_program,
        "selected_points": len(selected) * config.expected_points_per_program,
        "global_transfer_bandwidths": config.transfer_bandwidths.to_dict(),
        "selected_case_ids": [case.case_id for case in selected],
        "resume": arguments.resume,
        "resume_provenance": resume_provenance,
    }
    print(json.dumps(matrix, indent=2, sort_keys=True), flush=True)
    if arguments.dry_run:
        return 0
    if resume_directory is None:
        paths = BaselinePaths.initialize(
            arguments.output_dir,
            baseline_id=baseline_id,
            config=config,
            provenance=provenance,
            corpus_root=arguments.corpus_dir,
            corpus_digest=corpus_digest,
            cases=cases,
        )
    else:
        paths = BaselinePaths.open_existing(
            resume_directory,
            config=config,
            corpus_digest=corpus_digest,
        )
        _record_resume(paths, provenance, resume_provenance)
    launch_path = paths.directory / "launch-command.txt"
    if not launch_path.exists():
        atomic_text(
            launch_path,
            shlex.join(
                [
                    sys.executable,
                    "-m",
                    "benchmarking.planning_eval.evaluate",
                    *sys.argv[1:],
                ]
            )
            + "\n",
        )
    try:
        summary = run_frontier_collection(
            config,
            cases,
            selected,
            paths=paths,
            options=ControllerOptions(
                artifact_store=arguments.artifact_store.expanduser().resolve(),
                resume=arguments.resume,
                verbose_pressurefit=arguments.verbose_pressurefit,
                revision=arguments.revision or provenance.head,
            ),
            repository_root=repository_root,
        )
    except KeyboardInterrupt:
        print("frontier collection interrupted by user", flush=True)
        return 130
    failures = summary.get("case_failures")
    status_counts = summary.get("status_counts")
    has_errors = (
        isinstance(status_counts, dict)
        and int(status_counts.get("error", 0)) > 0
    )
    return 1 if failures or has_errors else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run PressureFit over a frozen Program corpus. Every point is "
            "atomically persisted and Program workers are crash-isolated."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--planning-cache", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--revision",
        help=(
            "record this revision on every point this run produces, and on "
            "--resume select the baseline recorded under it; defaults to the "
            "current HEAD. It labels the run - it does not check anything "
            "out, so the code that runs is whatever is in the worktree"
        ),
    )
    parser.add_argument("--case", action="append", default=[], metavar="GLOB")
    parser.add_argument("--start-at")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--verbose-pressurefit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _select_cases(
    cases: tuple[CorpusProgramCase, ...],
    *,
    patterns: tuple[str, ...],
    start_at: str | None,
    limit: int | None,
) -> tuple[CorpusProgramCase, ...]:
    selected = tuple(
        case
        for case in cases
        if not patterns
        or any(
            fnmatch.fnmatchcase(case.case_id, item)
            for item in patterns
        )
    )
    if patterns and not selected:
        raise ValueError("--case patterns selected no Program cases")
    if start_at is not None:
        matches = tuple(
            index
            for index, case in enumerate(selected)
            if case.case_id == start_at
        )
        if len(matches) != 1:
            raise ValueError(f"--start-at case {start_at!r} was not selected")
        selected = selected[matches[0] :]
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        selected = selected[:limit]
    if not selected:
        raise ValueError("no Program cases remain after filtering")
    return selected


def _find_resume_baseline(
    output_root: Path,
    *,
    baseline_id: str,
    config_digest: str,
    corpus_digest: str,
    provenance: RepositoryProvenance,
    enabled: bool,
    select_revision: str | None = None,
) -> tuple[Path | None, dict[str, object] | None]:
    if not enabled:
        return None, None
    output = output_root.expanduser().resolve()
    exact = output / baseline_id
    if select_revision is None and exact.is_dir():
        return exact, {
            "recorded_head": provenance.head,
            "resume_head": provenance.head,
            "changed_files": [],
            "classification": "exact_source",
            "spans_revisions": False,
        }
    matches: list[tuple[Path, dict[str, object]]] = []
    unreadable: list[str] = []
    for directory in sorted(output.glob("*")):
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = read_object(manifest_path)
        corpus = manifest.get("corpus")
        if manifest.get("config_digest") != config_digest:
            continue
        if (
            not isinstance(corpus, dict)
            or corpus.get("manifest_digest") != corpus_digest
        ):
            continue
        repository = manifest.get("repository")
        if select_revision is not None and (
            not isinstance(repository, dict)
            or not str(repository.get("head", "")).startswith(select_revision)
        ):
            continue
        summary_path = directory / "summary.json"
        if summary_path.is_file():
            pending = read_object(summary_path).get("pending_points")
            if pending == 0:
                continue
        try:
            relationship = resume_provenance_relationship(
                provenance,
                manifest.get("repository"),
            )
        except ValueError as error:
            unreadable.append(f"{directory.name}: {error}")
        else:
            matches.append((directory, relationship))
    if len(matches) > 1:
        raise ValueError(
            "--resume matched more than one baseline for this config and "
            "corpus; name one with --revision, or move the rest aside: "
            + ", ".join(path.name for path, _ in matches)
        )
    if matches:
        return matches[0]
    if unreadable:
        raise ValueError(
            "--resume found matching baselines whose provenance is unreadable: "
            + "; ".join(unreadable)
        )
    return None, None


def _record_resume(
    paths: BaselinePaths,
    provenance: RepositoryProvenance,
    compatibility: dict[str, object] | None,
) -> None:
    command = shlex.join(
        [
            sys.executable,
            "-m",
            "benchmarking.planning_eval.evaluate",
            *sys.argv[1:],
        ]
    )
    record = {
        "schema": "shadowspill.pressurefit_frontier_resume/v1",
        "started_at": utc_now(),
        "baseline_id": paths.directory.name,
        "repository": provenance.to_dict(),
        "compatibility": compatibility,
        "command": command,
    }
    append_log(
        paths.directory / "resume-history.jsonl",
        json.dumps(record, sort_keys=True, separators=(",", ":")),
    )
    append_log(paths.directory / "resume-commands.log", command)


__all__ = ["main"]
