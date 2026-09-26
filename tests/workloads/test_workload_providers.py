"""A workload family's operation providers, chosen for a block or from a point on."""

from __future__ import annotations

import contextvars

import pytest

pytest.importorskip("mlops")

from mlops.dispatch.context import implementation_override

from workloads.providers import (
    implementation_context,
    select_implementation,
)

OPERATIONS = ("embedding", "rms_norm", "rope", "flash_attention", "swiglu", "head_loss")


def _selected() -> dict[str, str | None]:
    return {operation: implementation_override(operation) for operation in OPERATIONS}


def test_selecting_a_family_from_a_point_on_selects_what_its_block_does() -> None:
    with implementation_context("llama3", "mlops"):
        in_block = _selected()

    def choose() -> dict[str, str | None]:
        select_implementation("llama3", "mlops")
        return _selected()

    assert all(in_block.values())
    assert contextvars.copy_context().run(choose) == in_block
    assert not any(_selected().values())  # the copied context kept it


def test_pytorch_selects_nothing_and_an_unknown_family_is_refused() -> None:
    def choose() -> dict[str, str | None]:
        select_implementation("llama3", "pytorch")
        return _selected()

    assert not any(contextvars.copy_context().run(choose).values())
    with pytest.raises(ValueError, match="unknown mlops model family"):
        select_implementation("gpt2", "mlops")
