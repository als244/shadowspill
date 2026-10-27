"""Build commands for the existing sweep and report entry points."""

from __future__ import annotations

import sys
from pathlib import Path


def sweep_command(
    spec: dict,
    sweep: dict,
    matrix: Path,
    output: Path,
    *,
    limit: int | None = None,
    retry_failed: bool = False,
):
    """The sweep launcher's command line for one sweep over its written
    matrix, resuming when its manifest exists."""

    command = [
        sys.executable,
        "-m",
        "baselines.sweep",
        "--engine",
        sweep["engine"],
        "--config",
        str(matrix),
        "--output",
        str(output),
        "--gpus",
        str(spec["gpus"]),
        "--python",
        sweep["python"],
        "--continue-on-error",
        "--case-timeout",
        str(sweep["case_timeout"]),
    ]
    if (output / "manifest.json").exists():
        command.append("--resume")
        if retry_failed:
            command.append("--retry-failed")
    if limit is not None:
        command.extend(["--limit", str(limit)])
    return command


def report_command(outputs: list[Path], report: Path):
    return [
        sys.executable,
        "-m",
        "baselines.report",
        *(str(path) for path in outputs),
        "--output",
        str(report),
        "--memory",
        "reserved",
    ]
