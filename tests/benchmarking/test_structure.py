from __future__ import annotations

import ast
from pathlib import Path

from benchmarking.data_geometry import DataGeometry

_REPOSITORY = Path(__file__).resolve().parents[2]
_BENCHMARKING = _REPOSITORY / "benchmarking"


def test_data_geometry_groups_primary_and_derived_axes() -> None:
    geometry = DataGeometry(1024, 8192, 8)

    assert geometry.to_dict() == {
        "sequence_length": 1024,
        "tokens_per_microbatch": 8192,
        "sequences_per_microbatch": 8,
        "accumulation_rounds": 8,
        "tokens_per_optimizer_step": 65536,
    }


def test_benchmarking_python_does_not_import_qualification() -> None:
    violations: list[str] = []
    for path in sorted(_BENCHMARKING.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            modules: tuple[str, ...] = ()
            if isinstance(node, ast.Import):
                modules = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                modules = (node.module,)
            if any(
                name == "qualification" or name.startswith("qualification.")
                for name in modules
            ):
                violations.append(str(path.relative_to(_REPOSITORY)))
    assert violations == []
