"""JSON configs: overrides by dotted key, and the forms that name objects."""

from __future__ import annotations

import datetime
import fractions
import json
from pathlib import Path

import pytest

from training import config


def _write(tmp_path: Path, value: object) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value))
    return path


def test_overrides_replace_values_by_dotted_key(tmp_path: Path) -> None:
    path = _write(tmp_path, {"steps": 10, "backend": {"execution_gib": 8}})
    loaded = config.load(
        path,
        ["steps=20", "backend.execution_gib=12.5", "name=plain words", "flags=[1, 2]"],
    )
    assert loaded == {
        "steps": 20,
        "backend": {"execution_gib": 12.5},
        "name": "plain words",
        "flags": [1, 2],
    }
    with pytest.raises(ValueError, match="key=value"):
        config.load(path, ["steps"])


def test_references_name_objects_calls_and_partials() -> None:
    resolved = config.resolve(
        {
            "constant": "@fractions:Fraction",
            "dotted": "@datetime:date.fromisoformat",
            "called": {"@call": "fractions:Fraction", "numerator": 1, "denominator": 3},
            "partial": {"@partial": "builtins:int", "base": 2},
            "nested": [{"@call": "fractions:Fraction", "numerator": 2}],
            "plain": {"a": 1},
        }
    )
    assert resolved["constant"] is fractions.Fraction
    assert resolved["dotted"] == datetime.date.fromisoformat
    assert resolved["called"] == fractions.Fraction(1, 3)
    assert resolved["partial"]("101") == 5
    assert resolved["nested"] == [fractions.Fraction(2)]
    assert resolved["plain"] == {"a": 1}
    with pytest.raises(ValueError, match="module:name"):
        config.resolve("@fractions")
