"""Qualification must build its own artifacts, without reading a user"s cache."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_suite_planning_calls_specify_artifact_stores() -> None:
    """Caches can be shared within a test, but must be rooted by that test."""
    missing = []
    for path in (ROOT / "tests").rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            name = (
                function.id if isinstance(function, ast.Name)
                else function.attr if isinstance(function, ast.Attribute)
                else None
            )
            if name not in {"plan_step", "plan_forward"}:
                continue
            stores = [
                keyword.value for keyword in node.keywords
                if keyword.arg in {"artifact_store", "build_store"}
            ]
            if not stores or any(
                isinstance(value, ast.Constant) and value.value is None
                for value in stores
            ):
                missing.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not missing, (
        "Suite planning needs a temporary artifact store: " + ", ".join(missing)
    )
