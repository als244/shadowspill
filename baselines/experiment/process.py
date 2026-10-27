"""Launch sweeps and reports with console logs and interruption cleanup."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

from baselines.common.provenance import ROOT

from .reporting import Console


def run_process(command, log: Path, dry_run: bool) -> int | None:
    """Run one launcher with its console in ``log``; nothing in a dry run."""

    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        handle.write(f"$ {shlex.join(command)}\n")
        handle.flush()
        if dry_run:
            return None
        return subprocess.run(
            command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT
        ).returncode


def run_sweep(command, log: Path, console: Console) -> int:
    """Run one sweep launcher to its end; every line of its console goes to
    ``log`` and, indented, to the experiment's."""

    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        handle.write(f"$ {shlex.join(command)}\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            for line in process.stdout:
                handle.write(line)
                handle.flush()
                console(line.rstrip("\n"))
        finally:
            returncode = process.wait()
    return returncode
