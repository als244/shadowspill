"""Tests for configurable and external qualification cases."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest
import torch.nn as nn

from qualification.numerical import cases


def test_builtin_case_accepts_model_config_and_data_geometry() -> None:
    case = cases.build_case(
        "llama3",
        model_config={
            "n_layers": 2,
            "d_model": 64,
            "n_heads": 4,
            "n_kv_heads": 2,
            "d_ff": 128,
            "vocab_size": 512,
            "max_seq_len": 32,
        },
        data_geometry=[
            {"token_shape": [1, 16], "sequence_lengths": [7, 9]},
            {"token_shape": [2, 12], "sequence_lengths": [8, 8, 8]},
        ],
    )

    assert case.model.config.n_layers == 2  # type: ignore[attr-defined]
    assert [tuple(item[0].shape) for item in case.microbatches] == [
        (1, 16),
        (2, 12),
    ]
    assert case.microbatches[1][2] == (8, 8, 8)


def test_external_factory_receives_complete_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class CustomCase:
        def __init__(self) -> None:
            self.family = "custom_decoder"
            self.model_implementation = "pytorch"
            self.model = nn.Linear(2, 2)
            self.microbatches: list[list[Any]] = [[1, "metadata"]]

        @staticmethod
        def objective(*arguments: object) -> object:
            return arguments

        @staticmethod
        def optimizer(*arguments: object) -> object:
            return arguments

        @staticmethod
        def implementations() -> object:
            return nullcontext()

    def factory(**values: object) -> CustomCase:
        observed.update(values)
        return CustomCase()

    monkeypatch.setattr(
        cases.importlib,
        "import_module",
        lambda name: SimpleNamespace(build=factory),
    )
    result = cases.build_case(
        "custom_decoder",
        seed=7,
        model_config={"width": 2},
        data_geometry=[{"shape": [3, 2]}],
        case_factory="example.cases:build",
        case_options={"objective": "custom"},
    )

    assert result.family == "custom_decoder"
    assert observed == {
        "model_name": "custom_decoder",
        "model_implementation": "pytorch",
        "seed": 7,
        "model_config": {"width": 2},
        "data_geometry": ({"shape": [3, 2]},),
        "case_options": {"objective": "custom"},
    }
