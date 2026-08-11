#!/usr/bin/env python3
"""Reject legacy implementation names from production source paths."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = (
    ROOT / "src" / "shadowspill",
    ROOT / "runtime",
    ROOT / "simulator",
    ROOT / "planner",
    ROOT / "pytorch",
)
TEXT_SUFFIXES = {".c", ".cc", ".cpp", ".h", ".hpp", ".py"}
FORBIDDEN = {
    "legacy project name": re.compile(r"\bdataflow\b", re.IGNORECASE),
    "legacy simulator prefix": re.compile(r"\bDFS_[A-Za-z0-9_]*\b"),
    "legacy C prefix": re.compile(r"\bdf_[A-Za-z0-9_]*\b"),
    "legacy Python namespace": re.compile(r"\bmlops_planning\b"),
}


def production_files() -> list[Path]:
    files: list[Path] = []
    for root in SOURCE_ROOTS:
        if not root.exists():
            continue
        files.extend(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix in TEXT_SUFFIXES
        )
    return sorted(files)


def main() -> int:
    failures: list[str] = []
    for path in production_files():
        text = path.read_text(encoding="utf-8")
        for label, pattern in FORBIDDEN.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                failures.append(
                    f"{path.relative_to(ROOT)}:{line}: {label}: {match.group(0)!r}"
                )
    if failures:
        print("\n".join(failures))
        return 1
    print(f"naming audit passed ({len(production_files())} production files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
