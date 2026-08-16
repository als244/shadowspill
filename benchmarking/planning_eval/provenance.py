"""Exact repository and environment identity for a planner baseline."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_HARNESS_ONLY_RESUME_PATHS = frozenset(
    {
        "benchmarking/planning_eval/README.md",
        "benchmarking/planning_eval/cli.py",
        "benchmarking/planning_eval/controller.py",
        "benchmarking/planning_eval/process.py",
        "benchmarking/planning_eval/provenance.py",
        "benchmarking/planning_eval/storage.py",
        "benchmarking/planning_eval/summary.py",
        "tests/benchmarking/test_planning_eval.py",
    }
)


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


def compatible_resume_provenance(
    current: RepositoryProvenance,
    recorded: object,
) -> dict[str, object]:
    """Prove that an older baseline differs only in orchestration evidence."""

    if not isinstance(recorded, dict):
        raise ValueError("baseline repository provenance is invalid")
    previous_head = recorded.get("head")
    previous_dirty = recorded.get("dirty")
    if not isinstance(previous_head, str) or len(previous_head) != 40:
        raise ValueError("baseline repository head is invalid")
    if previous_dirty is not False:
        raise ValueError("automatic cross-revision resume requires a clean baseline")
    if current.dirty:
        raise ValueError("automatic cross-revision resume requires a clean worktree")
    if not _is_ancestor(current.repository_root, previous_head, current.head):
        raise ValueError("baseline revision is not an ancestor of the current revision")
    changed = tuple(
        sorted(
            item
            for item in _git(
                current.repository_root,
                "diff",
                "--name-only",
                f"{previous_head}..{current.head}",
            ).splitlines()
            if item
        )
    )
    unsafe = tuple(item for item in changed if item not in _HARNESS_ONLY_RESUME_PATHS)
    if unsafe:
        raise ValueError(
            "planner-affecting files changed since the baseline: "
            + ", ".join(unsafe)
        )
    return {
        "recorded_head": previous_head,
        "resume_head": current.head,
        "changed_files": list(changed),
        "classification": "harness_only",
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


def _is_ancestor(root: Path, ancestor: str, descendant: str) -> bool:
    completed = subprocess.run(
        ("git", "merge-base", "--is-ancestor", ancestor, descendant),
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


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
    "compatible_resume_provenance",
    "environment_provenance",
]
