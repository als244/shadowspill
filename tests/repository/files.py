"""Source files in this checkout, excluding ignored experiments and outputs."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def repository_files(pattern: str) -> tuple[Path, ...]:
    """Include new files and omit deleted files while edits are in progress."""

    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
            pattern,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(
        ROOT / name
        for name in sorted(set(result.stdout.split("\0")) - {""})
        if (ROOT / name).is_file()
    )
