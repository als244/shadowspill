"""Frozen candidate descriptions shared by every saved phase."""

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

from shadowspill.orchestration.launch.workers import Workers


def build_candidates(options: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    return tuple(options["candidates"])


def failures(root: Path) -> dict[str, Any]:
    path = root / "candidate-failures.json"
    return cast(dict[str, Any], json.loads(path.read_text())) if path.exists() else {}


def available(root: Path, options: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    rejected = failures(root)
    return tuple(c for c in build_candidates(options) if c["path"] not in rejected)


def attempt_candidate[T](
    root: Path,
    candidate: Mapping[str, Any],
    phase: str,
    operation: Callable[[], T],
    group: Workers,
) -> T | None:
    """Skip only local device exhaustion; a collective failure ends the group."""
    if group.count > 1:
        return operation()
    from shadowspill.orchestration.candidates import attempt

    entry: dict[str, Any] = {"name": candidate["path"]}
    result = attempt(entry, phase, operation, device=0)
    if "failure" in entry:
        rejected = failures(root)
        rejected[candidate["path"]] = entry["failure"]
        (root / "candidate-failures.json").write_text(
            json.dumps(rejected, indent=2) + "\n"
        )
    return result


def program_paths(
    options: Mapping[str, Any], *, root: Path | None = None
) -> tuple[str, ...]:
    kinds = ("training", "evaluation") if options["eval_every"] else ("training",)
    return tuple(
        f"{candidate['path']}/{kind}.ssp"
        for candidate in (
            build_candidates(options) if root is None else available(root, options)
        )
        for kind in kinds
    )
