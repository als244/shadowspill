"""CLI for reproducible, resumable PressureFit frontier baselines."""

from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path

from .config import load_frontier_config
from .controller import ControllerOptions, run_frontier_collection
from .provenance import capture_repository_provenance
from .source import (
    CorpusProgramCase,
    corpus_manifest_digest,
    discover_program_cases,
)
from .storage import BaselinePaths


def main() -> int:
    parser = _parser()
    arguments = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[3]
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
    baseline_id = provenance.baseline_id(config.name, config.digest)
    matrix = {
        "baseline_id": baseline_id,
        "config_digest": config.digest,
        "corpus_manifest_digest": corpus_manifest_digest(cases),
        "total_programs": len(cases),
        "selected_programs": len(selected),
        "points_per_program": config.expected_points_per_program,
        "total_points": len(cases) * config.expected_points_per_program,
        "selected_points": len(selected) * config.expected_points_per_program,
        "global_transfer_bandwidths": config.transfer_bandwidths.to_dict(),
        "selected_case_ids": [case.case_id for case in selected],
    }
    print(json.dumps(matrix, indent=2, sort_keys=True), flush=True)
    if arguments.dry_run:
        return 0
    paths = BaselinePaths.initialize(
        arguments.output_dir,
        baseline_id=baseline_id,
        config=config,
        provenance=provenance,
        corpus_root=arguments.corpus_dir,
        corpus_digest=corpus_manifest_digest(cases),
        cases=cases,
    )
    try:
        summary = run_frontier_collection(
            config,
            cases,
            selected,
            paths=paths,
            options=ControllerOptions(
                planning_cache=arguments.planning_cache.expanduser().resolve(),
                resume=arguments.resume,
                verbose_pressurefit=arguments.verbose_pressurefit,
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


__all__ = ["main"]
