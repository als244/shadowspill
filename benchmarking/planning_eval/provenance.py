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

    @property
    def untracked_sources(self) -> tuple[str, ...]:
        """Untracked files under src/ or csrc/, which the diff cannot capture."""

        return tuple(
            line[3:]
            for line in self.status.splitlines()
            if line.startswith("?? ") and line[3:].startswith(("src/", "csrc/"))
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "repository_root": str(self.repository_root),
            "head": self.head,
            "dirty": self.dirty,
            "status": self.status.splitlines(),
            "tracked_diff_sha256": self.diff_sha256,
            # A tracked diff cannot reproduce these, so the baseline says which
            # files it could not describe rather than refusing to start.
            "untracked_sources": list(self.untracked_sources),
        }


def capture_repository_provenance(root: Path) -> RepositoryProvenance:
    """Capture the exact source state used by worker subprocesses."""

    repository = root.expanduser().resolve()
    head = _git(repository, "rev-parse", "HEAD").strip()
    status = _git(repository, "status", "--porcelain=v1", "--untracked-files=all")
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


def resume_provenance_relationship(
    current: RepositoryProvenance,
    recorded: object,
) -> dict[str, object]:
    """Describe how a baseline's source relates to the current worktree.

    Resume never refuses over a revision. A run that spans revisions is a fact
    about the artifact, not a reason to make someone replay fifteen hours of
    planning, so this classifies the relationship and lets the baseline record
    it. Every point carries the revision that produced it, so a reader can see
    exactly where a baseline changed underneath itself.

    What still has to match is what is being measured: the frontier config and
    the corpus. Those are checked separately, and they do refuse.
    """

    if not isinstance(recorded, dict):
        raise ValueError("baseline repository provenance is invalid")
    previous_head = recorded.get("head")
    if not isinstance(previous_head, str) or len(previous_head) != 40:
        raise ValueError("baseline repository head is invalid")
    previous_dirty = bool(recorded.get("dirty"))

    if previous_head == current.head and not previous_dirty and not current.dirty:
        return {
            "recorded_head": previous_head,
            "resume_head": current.head,
            "changed_files": [],
            "classification": "exact_source",
            "spans_revisions": False,
        }
    changed: tuple[str, ...] = ()
    related = _is_ancestor(current.repository_root, previous_head, current.head)
    if related:
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
    if previous_dirty or current.dirty:
        classification = "dirty_worktree"
    elif not related:
        classification = "unrelated_revision"
    elif all(item in _HARNESS_ONLY_RESUME_PATHS for item in changed):
        classification = "harness_only"
    else:
        classification = "planner_changed"
    return {
        "recorded_head": previous_head,
        "resume_head": current.head,
        "changed_files": list(changed),
        "classification": classification,
        "spans_revisions": True,
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


__all__ = [
    "RepositoryProvenance",
    "capture_repository_provenance",
    "environment_provenance",
    "resume_provenance_relationship",
]
