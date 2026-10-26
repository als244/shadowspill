"""The supplied text recipe composes the generic trainer, including exact resume."""

from __future__ import annotations

import pytest
import torch

from tests.training._synthetic import NOTES
from tests.training._text_recipe import _config, _records
from workloads.recipes.text.run import run_config


@pytest.mark.parametrize("masters", [False, True])
def test_recipe_preserves_data_schedules_and_weights_on_explicit_resume(
    tmp_path, masters
):
    path = _config(tmp_path)
    precision = (
        [
            "model.dtype=bfloat16",
            "master_dtype=@torch:float32",
            "grad_dtype=@torch:float32",
        ]
        if masters
        else []
    )
    whole, resumed = tmp_path / "whole", tmp_path / "resumed"
    NOTES.clear()
    last = run_config(path, [f"run_dir={whole}", *precision])
    assert NOTES == ["first", "second"]
    assert last.step == 6
    losses = _records(whole, "train/loss")
    assert list(losses) == [1, 2, 3, 4, 5, 6]
    assert losses[6]["train/loss"] < losses[1]["train/loss"]
    assert list(_records(whole, "eval/loss")) == [3, 6]
    packing = _records(whole, "packing/trained_tokens")
    for step, row in packing.items():
        assert row["train/tokens_per_second"] == (
            row["packing/trained_tokens"] / losses[step]["train/step_seconds"]
        )
    run_config(path, [f"run_dir={resumed}", "steps=3", *precision])
    checkpoint = resumed / "checkpoints" / "step_00000003"
    run_config(path, [f"run_dir={resumed}", f"resume={checkpoint}", *precision])
    again = _records(resumed, "train/loss")
    for step in losses:
        for name in ("train/loss", "hyperparameters/lr"):
            assert again[step][name] == losses[step][name]
    assert (whole / "packing.jsonl").read_text() == (
        resumed / "packing.jsonl"
    ).read_text()

    def state(root):
        return torch.load(
            root / "checkpoints" / "step_00000006" / "state.pt", weights_only=True
        )

    expected, actual = state(whole), state(resumed)
    for name, value in expected["model"].items():
        assert torch.equal(value, actual["model"][name])
        if masters:
            assert value.dtype == torch.float32


def test_recipe_logs_tensor_observations_with_real_parameter_names(tmp_path):
    path = _config(tmp_path)
    root = tmp_path / "observed"
    run_config(
        path,
        [
            f"run_dir={root}",
            "steps=2",
            "eval_every=0",
            "checkpoint_every=0",
            "parameter_metrics=@shadowspill.training.observations:parameter_norms",
        ],
    )
    rows = _records(root, "train/grad_norm/global/l2")
    assert list(rows) == [1, 2]
    for row in rows.values():
        assert row["train/grad_rms/embed/weight"] > 0
        assert row["train/param_rms/embed/weight"] > 0
        assert row["train/grad_weight_ratio/head/weight"] > 0
        assert not any(
            name.startswith(("train/grad_squared_share/", "train/param_norm/"))
            for name in row
        )
    assert rows[2]["train/elapsed_seconds"] > rows[1]["train/elapsed_seconds"]


def test_recipe_rejects_invalid_text_geometry(tmp_path):
    path = _config(tmp_path)
    with pytest.raises(ValueError, match="whole fixed-length"):
        run_config(path, ["max_tokens_per_step=1000"])
    with pytest.raises(ValueError, match="no text microbatch"):
        run_config(path, ["max_tokens_per_microbatch=3072"])
