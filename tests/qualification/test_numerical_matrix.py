"""Tests for the reusable model-correctness matrix launcher."""

from __future__ import annotations

import argparse

import pytest

from qualification.numerical.matrix import _budget_overrides, _parse_bytes
from qualification.numerical.run import _case_identity


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1024", 1024),
        ("2_KiB", 2 << 10),
        ("10MiB", 10 << 20),
        ("16GiB", 16 << 30),
    ],
)
def test_parse_bytes(value: str, expected: int) -> None:
    assert _parse_bytes(value) == expected


@pytest.mark.parametrize("value", ["", "0", "-1", "1GB", "1.5GiB"])
def test_parse_bytes_rejects_invalid_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_bytes(value)


def test_family_budget_overrides_are_independent() -> None:
    assert _budget_overrides(
        ["llama3=10GiB", "qwen35=12GiB"],
        valid_models={"llama3", "qwen35"},
    ) == {
        "llama3": 10 << 30,
        "qwen35": 12 << 30,
    }


def test_budget_override_rejects_unknown_family() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _budget_overrides(["unknown=10GiB"], valid_models={"llama3"})


def test_case_identity_covers_model_and_data_configuration() -> None:
    common = {
        "model_name": "llama3",
        "model_implementation": "pytorch",
        "seed": 7,
        "model_config": {"n_layers": 2},
        "data_geometry": [{"token_shape": [1, 16]}],
        "case_factory": None,
        "case_options": {},
    }
    first = _case_identity(**common)  # type: ignore[arg-type]
    changed = dict(common)
    changed["data_geometry"] = [{"token_shape": [1, 32]}]
    assert _case_identity(**changed) != first  # type: ignore[arg-type]
