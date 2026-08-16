"""Launch the five ShadowSpill-only full-model qualification cells."""

from __future__ import annotations

import argparse
import json
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from models.full_model import manifests

_PLAN_PROGRESS = re.compile(
    r"^\[shadowspill\.plan \+\s*[0-9.]+s\]\s+"
    r"(?P<phase>[A-Za-z0-9_]+): "
    r"(?P<state>started|finished|failed)(?:\s+in\s+.*)?$"
)


def _active_planning_phases(log_text: str) -> tuple[str, ...]:
    """Recover the open phase stack from verbose planning output."""

    active: list[str] = []
    for line in log_text.splitlines():
        match = _PLAN_PROGRESS.match(line)
        if match is None:
            continue
        phase = match.group("phase")
        if match.group("state") == "started":
            active.append(phase)
            continue
        for index in range(len(active) - 1, -1, -1):
            if active[index] == phase:
                del active[index:]
                break
    return tuple(active)


def _termination_signal(return_code: int) -> str | None:
    if return_code >= 0:
        return None
    try:
        return signal.Signals(-return_code).name
    except ValueError:
        return f"signal_{-return_code}"


def _write_parent_failure(
    *,
    manifest_identity: str,
    return_code: int,
    log: Path,
    failure_path: Path,
) -> dict[str, object]:
    """Record failures a killed child had no opportunity to serialize."""

    text = log.read_text(errors="replace") if log.is_file() else ""
    active = _active_planning_phases(text)
    signal_name = _termination_signal(return_code)
    phase = active[-1] if active else None
    if signal_name is not None:
        message = f"qualification subprocess terminated by {signal_name}"
    else:
        message = f"qualification subprocess exited with status {return_code}"
    if phase is not None:
        message += f" during planning phase {phase!r}"
    message += "; no Python traceback was available"
    progress_lines = [
        line for line in text.splitlines() if line.startswith("[shadowspill.plan ")
    ]
    failure: dict[str, object] = {
        "schema": "shadowspill.full_model_subprocess_failure/v1",
        "identity": manifest_identity,
        "error_type": "SubprocessTermination",
        "error": message,
        "return_code": return_code,
        "termination_signal": signal_name,
        "active_planning_phases": active,
        "last_planning_progress": progress_lines[-1] if progress_lines else None,
        "log": str(log),
    }
    failure_path.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
    return failure


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("qualification/results/full_model"),
    )
    parser.add_argument("--force-fresh", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--implementation-revision")
    parser.add_argument(
        "--cells",
        nargs="*",
        help="optional identities such as mlops_llama3 or pytorch_qwen35",
    )
    arguments = parser.parse_args()
    output = arguments.output_directory.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected = set(arguments.cells or ())
    rows: list[dict[str, object]] = []
    failed = False
    for manifest in manifests():
        if selected and manifest.identity not in selected:
            continue
        artifact = output / f"{manifest.identity}.json"
        failure_path = artifact.with_suffix(".failure.json")
        # A rerun must never be classified from an artifact left by an older run.
        artifact.unlink(missing_ok=True)
        failure_path.unlink(missing_ok=True)
        command = [
            sys.executable,
            "-m",
            "qualification.performance.run",
            manifest.family,
            manifest.implementation,
            str(artifact),
            "--planning-cachedir",
            str(output / "planning_cache" / manifest.identity),
        ]
        if arguments.force_fresh:
            command.append("--force-fresh")
        if arguments.plan_only:
            command.append("--plan-only")
        if arguments.implementation_revision is not None:
            command.extend(
                ("--implementation-revision", arguments.implementation_revision)
            )
        log = output / f"{manifest.identity}.log"
        started = time.perf_counter()
        with log.open("w") as destination:
            process = subprocess.run(
                command,
                stdout=destination,
                stderr=subprocess.STDOUT,
                check=False,
            )
        elapsed = time.perf_counter() - started
        passed = False
        if artifact.is_file():
            passed = bool(json.loads(artifact.read_text()).get("passed"))
        failure_record: dict[str, object] | None = None
        if failure_path.is_file():
            failure_record = json.loads(failure_path.read_text())
        elif process.returncode != 0 and not artifact.is_file():
            failure_record = _write_parent_failure(
                manifest_identity=manifest.identity,
                return_code=process.returncode,
                log=log,
                failure_path=failure_path,
            )
        row = {
            "identity": manifest.identity,
            "return_code": process.returncode,
            "passed": passed,
            "elapsed_seconds": elapsed,
            "artifact": str(artifact),
            "artifact_exists": artifact.is_file(),
            "failure_artifact": (
                str(failure_path) if failure_record is not None else None
            ),
            "failure": failure_record,
            "log": str(log),
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if process.returncode != 0 or not passed:
            failed = True
            if not arguments.keep_going:
                break
    summary = {
        "schema": "shadowspill.full_model_matrix/v1",
        "plan_only": arguments.plan_only,
        "cells": rows,
        "passed": bool(rows) and not failed and all(row["passed"] for row in rows),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
