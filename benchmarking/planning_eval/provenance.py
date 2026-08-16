"""Exact repository and environment identity for a planner baseline."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RepositoryProvenance:
    """Committed revision plus the complete tracked working-tree patch."""

    repository_root: Path
    head: str
    status: str
    diff: str
    diff_sha256: str

    @property
    def dirty(self) -> bool:
        return bool(self.status.strip())

    def baseline_id(self, name: str, config_digest: str) -> str:
        state = f"dirty-{self.diff_sha256[:12]}" if self.dirty else "clean"
        return f"{name}__{self.head[:12]}__{state}__cfg-{config_digest[:12]}"

    def to_dict(self) -> dict[str, object]:
        return {
            "repository_root": str(self.repository_root),
            "head": self.head,
            "dirty": self.dirty,
            "status": self.status.splitlines(),
            "tracked_diff_sha256": self.diff_sha256,
        }


def capture_repository_provenance(root: Path) -> RepositoryProvenance:
    """Capture the exact source state used by worker subprocesses."""

    repository = root.expanduser().resolve()
    head = _git(repository, "rev-parse", "HEAD").strip()
    status = _git(repository, "status", "--porcelain=v1", "--untracked-files=all")
    _reject_untracked_runtime_sources(status)
    diff = _git(repository, "diff", "--binary", "HEAD")
    return RepositoryProvenance(
        repository,
        head,
        status,
        diff,
        hashlib.sha256(diff.encode()).hexdigest(),
    )


def environment_provenance() -> dict[str, object]:
    """Return enough process identity to reproduce the harness invocation."""

    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "pid": os.getpid(),
    }


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _reject_untracked_runtime_sources(status: str) -> None:
    protected = ("src/", "planner/", "runtime/", "simulator/")
    offenders = tuple(
        line[3:]
        for line in status.splitlines()
        if line.startswith("?? ") and line[3:].startswith(protected)
    )
    if offenders:
        raise ValueError(
            "frontier provenance cannot freeze untracked runtime source files: "
            + ", ".join(offenders)
        )


__all__ = [
    "RepositoryProvenance",
    "capture_repository_provenance",
    "environment_provenance",
]
