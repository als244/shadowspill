#!/usr/bin/env python3
"""Enforce ShadowSpill's production naming and provider boundaries."""

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
NEUTRAL_ROOTS = (
    ROOT / "src" / "shadowspill" / "ir",
    ROOT / "src" / "shadowspill" / "planner",
    ROOT / "src" / "shadowspill" / "runtime",
    ROOT / "src" / "shadowspill" / "simulator",
    ROOT / "runtime" / "include",
    ROOT / "runtime" / "src",
    ROOT / "planner" / "include",
    ROOT / "planner" / "src",
    ROOT / "simulator" / "include",
    ROOT / "simulator" / "src",
)
TEXT_SUFFIXES = {".c", ".cc", ".cpp", ".h", ".hpp", ".py"}
FORBIDDEN = {
    "legacy project name": re.compile(r"\bdataflow\b", re.IGNORECASE),
    "legacy simulator prefix": re.compile(r"\bDFS_[A-Za-z0-9_]*\b"),
    "legacy C prefix": re.compile(r"\bdf_[A-Za-z0-9_]*\b"),
    "legacy Python namespace": re.compile(r"\bmlops_planning\b"),
}
NEUTRAL_FORBIDDEN = {
    "provider name in neutral code": re.compile(
        r"\b(?:cuda|rocm|hip)\b", re.IGNORECASE
    ),
}
PRODUCTION_FORBIDDEN = {
    "old secondary-pool role": re.compile(r"\bbacking(?:_[A-Za-z0-9_]+)?\b"),
    "physical transfer direction used as policy": re.compile(
        r"\b(?:h2d|d2h|host_to_device|device_to_host)\b", re.IGNORECASE
    ),
    "old worker terminology": re.compile(
        r"\b(?:progress_thread|progress_main|progress_completions)\b",
        re.IGNORECASE,
    ),
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


def files_under(roots: tuple[Path, ...]) -> list[Path]:
    return sorted(
        path
        for root in roots
        if root.exists()
        for path in root.rglob("*")
        if path.is_file() and path.suffix in TEXT_SUFFIXES
    )


def collect_matches(
    paths: list[Path], patterns: dict[str, re.Pattern[str]]
) -> list[str]:
    failures: list[str] = []
    for path in paths:
        source = path.read_text(encoding="utf-8")
        for label, pattern in patterns.items():
            for match in pattern.finditer(source):
                line = source.count("\n", 0, match.start()) + 1
                failures.append(
                    f"{path.relative_to(ROOT)}:{line}: {label}: {match.group(0)!r}"
                )
    return failures


def main() -> int:
    production = production_files()
    failures = collect_matches(production, FORBIDDEN)
    failures.extend(collect_matches(production, PRODUCTION_FORBIDDEN))
    failures.extend(collect_matches(files_under(NEUTRAL_ROOTS), NEUTRAL_FORBIDDEN))
    if failures:
        print("\n".join(failures))
        return 1
    print(f"naming audit passed ({len(production)} production files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
