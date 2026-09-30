"""Workloads leave operation selection to mlops and retain caller overrides."""

from __future__ import annotations

import pytest

pytest.importorskip("mlops")

import torch
from mlops.dispatch import use_implementations
from mlops.dispatch.context import deterministic_required, implementation_override

from workloads.numerical import NumericalCase


def test_numerical_defaults_do_not_pin_operation_implementations() -> None:
    case = NumericalCase("llama3", "mlops", torch.nn.Identity(), [])
    with case.implementations(deterministic=True):
        assert deterministic_required()
        assert implementation_override("flash_attention") is None
    assert not deterministic_required()


def test_numerical_context_keeps_an_explicit_caller_selection() -> None:
    case = NumericalCase("llama3", "mlops", torch.nn.Identity(), [])
    choice = "native_torch.flash_attention"
    with use_implementations({"flash_attention": choice}), case.implementations():
        assert implementation_override("flash_attention") == choice
    assert implementation_override("flash_attention") is None
