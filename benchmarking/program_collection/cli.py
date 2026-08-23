"""Command-line interface for resumable Program corpus collection."""

from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
from pathlib import Path

from .config import load_collection_config
from .controller import ControllerOptions, run_collection
from .matrix import ProgramRequest, expand_program_requests


def _head(root: Path) -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def main() -> int:
    parser = _parser()
    arguments = parser.parse_args()
    config = load_collection_config(arguments.config.expanduser().resolve())
    requests = expand_program_requests(config)
    selected = _select_requests(
        requests,
        patterns=tuple(arguments.case),
        start_at=arguments.start_at,
        limit=arguments.limit,
    )
    timeout = (
        config.case_timeout_seconds
        if arguments.timeout_seconds is None
        else arguments.timeout_seconds
    )
    max_attempts = (
        config.max_attempts
        if arguments.max_attempts is None
        else arguments.max_attempts
    )
    if timeout <= 0:
        parser.error("--timeout-seconds must be positive")
    if max_attempts <= 0:
        parser.error("--max-attempts must be positive")
    matrix = _matrix_summary(config.digest, requests, selected)
    print(json.dumps(matrix, indent=2, sort_keys=True), flush=True)
    if arguments.dry_run:
        return 0
    options = ControllerOptions(
        revision=arguments.revision or _head(Path.cwd()),
        planning_cache=arguments.planning_cache.expanduser().resolve(),
        resume=arguments.resume,
        timeout_seconds=timeout,
        max_attempts=max_attempts,
        quiet_plan=arguments.quiet_plan,
        force_fresh=arguments.force_fresh,
    )
    try:
        summary = run_collection(
            config,
            requests,
            selected,
            output_root=arguments.output_dir,
            options=options,
        )
    except KeyboardInterrupt:
        print("collection interrupted by user", flush=True)
        return 130
    raw_failed = summary["failed_cases"]
    if not isinstance(raw_failed, list):
        raise TypeError("collection summary failed_cases is not a list")
    failed = {str(item) for item in raw_failed}
    selected_ids = {request.case_id for request in selected}
    return 1 if failed & selected_ids else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect reusable pre-PressureFit Programs in isolated subprocesses. "
            "Every case failure is journaled and collection continues."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--planning-cache", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="validate and skip completed Programs, then retry incomplete cases",
    )
    parser.add_argument(
        "--revision",
        help=(
            "record this revision on every case this run produces; defaults "
            "to the current HEAD. It labels the run - it does not check "
            "anything out"
        ),
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        metavar="GLOB",
        help="collect only matching case IDs; repeat for multiple patterns",
    )
    parser.add_argument(
        "--start-at",
        help="start at one exact case ID after filtering",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout-seconds", type=int)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--quiet-plan", action="store_true")
    parser.add_argument(
        "--force-fresh",
        action="store_true",
        help="bypass and replace planning-cache artifacts for worker cases",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate config and print the selected matrix without collecting",
    )
    return parser


def _select_requests(
    requests: tuple[ProgramRequest, ...],
    *,
    patterns: tuple[str, ...],
    start_at: str | None,
    limit: int | None,
) -> tuple[ProgramRequest, ...]:
    selected = tuple(
        request
        for request in requests
        if not patterns
        or any(fnmatch.fnmatchcase(request.case_id, pattern) for pattern in patterns)
    )
    if patterns and not selected:
        raise ValueError("--case patterns selected no Program requests")
    if start_at is not None:
        indices = tuple(
            index
            for index, request in enumerate(selected)
            if request.case_id == start_at
        )
        if len(indices) != 1:
            raise ValueError(f"--start-at case {start_at!r} was not selected")
        selected = selected[indices[0] :]
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        selected = selected[:limit]
    if not selected:
        raise ValueError("no Program requests remain after filtering")
    return selected


def _matrix_summary(
    config_digest: str,
    requests: tuple[ProgramRequest, ...],
    selected: tuple[ProgramRequest, ...],
) -> dict[str, object]:
    by_model: dict[str, int] = {}
    for request in requests:
        by_model[request.model.name] = by_model.get(request.model.name, 0) + 1
    return {
        "config_digest": config_digest,
        "total_programs": len(requests),
        "selected_programs": len(selected),
        "programs_by_model": by_model,
        "selected_case_ids": [request.case_id for request in selected],
    }


__all__ = ["main"]
