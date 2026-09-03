"""Tests for the reusable model-correctness matrix launcher."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from tools.qualification.numerical import _case_identity
from tools.qualification.numerical_matrix import (
    _DEFAULT_IMPLEMENTATIONS,
    _budget_overrides,
    _parse_bytes,
)
from tools.qualification.references import (
    DEFAULT_APPROXIMATELY_1B_REFERENCE_DIRECTORY,
    canonical_reference_path,
    reference_artifact_exists,
    reference_inputs_path,
)
from workloads.numerical import DEFAULT_DEVICE_BUDGETS


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


def test_default_gate_is_five_cells_with_olmoe_under_pressure() -> None:
    assert sum(len(values) for values in _DEFAULT_IMPLEMENTATIONS.values()) == 5
    assert _DEFAULT_IMPLEMENTATIONS["olmoe"] == ("mlops",)
    assert DEFAULT_DEVICE_BUDGETS["olmoe"] == 8 << 30


def test_default_gate_reads_the_repo_local_reference_set() -> None:
    assert (
        Path("qualification/results/references/approximately_1b")
        == DEFAULT_APPROXIMATELY_1B_REFERENCE_DIRECTORY
    )


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


def test_canonical_reference_path_is_grouped_by_model_and_provider(
    tmp_path: Path,
) -> None:
    assert (
        canonical_reference_path(
            tmp_path,
            model_name="qwen35",
            implementation="mlops",
        )
        == tmp_path / "qwen35" / "mlops" / "reference.pt"
    )


def test_reference_artifact_requires_state_and_exact_inputs(tmp_path: Path) -> None:
    reference = tmp_path / "reference.pt"
    reference.touch()
    assert not reference_artifact_exists(reference)
    reference_inputs_path(reference).touch()
    assert reference_artifact_exists(reference)
