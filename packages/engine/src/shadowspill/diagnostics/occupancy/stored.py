"""Locate the IR paired with a low-level planner selection."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def program_path_for(selection_path: Path, selection: Mapping[str, Any]) -> Path:
    """Where the store keeps the program a selection was made for."""

    digest = str(selection["program_digest"])
    # <store>/vN/planning/results/<xx>/<key>/selection.json -> <store>/vN/planning
    planning = selection_path.resolve().parents[3]
    return planning / "programs" / digest[:2] / digest / "program.json"
