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


CONFIGS = Path(__file__).resolve().parents[2] / "training" / "configs"


@pytest.mark.parametrize(
    "path", sorted(CONFIGS.glob("*.json")), ids=lambda path: path.stem
)
def test_a_config_on_mlops_asks_it_for_weight_gradients_at_its_grad_dtype(
    path: Path,
) -> None:
    """mlops kernels return the weight gradients they sum unrounded only when
    asked at the dtype gradients are kept at; a config keeping them at the
    weights' own asks for nothing."""

    raw = json.loads(path.read_text())
    settings = raw.get("settings", [])
    if not any(
        setting.get("@call") == "workloads.providers:select_implementation"
        and setting.get("implementation") == "mlops"
        for setting in settings
    ):
        pytest.skip("the model does not run on mlops kernels")
    asked = [
        setting.get("dtype")
        for setting in settings
        if setting.get("@call") == "mlops.dispatch:set_weight_gradient_dtype"
    ]
    assert asked == ([raw["grad_dtype"]] if "grad_dtype" in raw else [])
