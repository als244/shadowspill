"""The config runner refuses keys it does not know, expands a checkpoint entry
into the model and its initializer, and makes a scheduled number a tensor."""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from shadowspill.training import config
from tests.workloads.test_checkpoints import huggingface_state, materialized, metadata
from workloads.recipes.text.experiment.hyperparameters import scheduled_arguments
from workloads.recipes.text.run import expand_checkpoint, load_config, run_config


def test_an_unknown_key_is_refused_with_the_nearest_name(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"steps": 3, "eval_evry": 2, "run_dir": str(tmp_path)}))
    with pytest.raises(ValueError, match=r"eval_evry \(did you mean eval_every\?\)"):
        run_config(path)


def test_a_renamed_key_is_refused_with_its_new_name(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"eval_microbatches": 2, "run_dir": str(tmp_path)}))
    with pytest.raises(ValueError, match=r"eval_microbatches \(now eval_tokens"):
        run_config(path)


def test_a_checkpoint_entry_stands_for_the_model_and_its_initializer(tmp_path):
    family, name = "llama3", "llama3-tiny"
    source = materialized(name)
    directory = tmp_path / "checkpoint"
    directory.mkdir()
    save_file(
        huggingface_state(family, source, source.config),
        directory / "model.safetensors",
    )
    (directory / "config.json").write_text(json.dumps(metadata(family, source.config)))
    raw = {
        "checkpoint": {
            "family": family,
            "directory": str(directory),
            "dtype": "bfloat16",
            "lora": {"rank": 4, "alpha": 8},
            "seed": 3,
        },
        "max_seq_len": 64,
        "seed": 0,
    }
    expanded = expand_checkpoint(raw)
    assert expanded["checkpoint"] == raw["checkpoint"]
    assert expanded["model"] == {
        "@call": "workloads.models.catalog:build_from_checkpoint",
        "family": family,
        "directory": str(directory),
        "max_seq_len": 64,
        "dtype": "bfloat16",
        "lora": {"rank": 4, "alpha": 8},
    }
    assert expanded["initialize"] == {
        "@call": "workloads.recipes.text.initialization:from_checkpoint",
        "family": family,
        "directory": str(directory),
        "max_seq_len": 64,
        "seed": 3,
    }
    model = config.resolve(expanded["model"])
    with config.resolve(expanded["initialize"]) as initialize:
        assert initialize.config == model.config and initialize.seed == 3
    assert any(name.endswith("lora_a") for name, _ in model.named_parameters())
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())
    # the initializer's seed follows the run's when the entry gives none
    plain = expand_checkpoint(
        {"checkpoint": {"family": family, "directory": "d"}, "seed": 7}
    )
    assert plain["initialize"]["seed"] == 7 and "max_seq_len" not in plain["model"]
    with pytest.raises(ValueError, match="one form or the other"):
        expand_checkpoint({**raw, "model": {}})
    with pytest.raises(ValueError, match="family and directory"):
        expand_checkpoint({"checkpoint": {"family": family}})
    assert expand_checkpoint({"steps": 1}) == {"steps": 1}


def test_a_number_a_schedule_names_becomes_a_tensor():
    schedules = {"optimizer_group.0.lr": object()}
    arguments = scheduled_arguments(
        {"lr": 1e-5, "betas": [0.9, 0.95], "eps": 1e-8}, schedules
    )
    assert isinstance(arguments["lr"], torch.Tensor)
    assert arguments["lr"].dtype == torch.float32 and arguments[
        "lr"
    ].item() == pytest.approx(1e-5)
    assert arguments["betas"] == [0.9, 0.95] and arguments["eps"] == 1e-8
    assert scheduled_arguments({"lr": 1e-5}, {}) == {"lr": 1e-5}
    held = torch.tensor(2e-5)
    assert scheduled_arguments({"lr": held}, schedules)["lr"] is held


def test_defaults_fill_what_a_config_leaves_out(tmp_path):
    raw = {
        "checkpoint": {"family": "llama3", "directory": "/models/x"},
        "data": "data/x",
        "optimizer_args": {"lr": 1e-5},
        "max_seq_len": 64,
        "max_tokens_per_step": 128,
        "steps": 2,
    }
    expanded = load_config(raw, ["run_dir=runs/x"])
    assert expanded["data"] == {
        "@call": "workloads.recipes.text.data.packed:PackedTokens",
        "directory": "data/x",
    }
    assert (
        expanded["objective"]
        == "@workloads.recipes.text.objectives:next_token_prediction"
    )
    assert expanded["optimizer"] == "@mlops.optim:AdamW"
    assert expanded["model"]["@call"].endswith("build_from_checkpoint")
    assert raw["data"] == "data/x"  # the caller's mapping is untouched
    with pytest.raises(
        ValueError, match="a config needs max_seq_len, max_tokens_per_step"
    ):
        load_config(
            {"checkpoint": raw["checkpoint"], "data": "d", "steps": 1}, ["run_dir=r"]
        )
    with pytest.raises(ValueError, match="names a checkpoint or a model"):
        load_config({k: v for k, v in raw.items() if k != "checkpoint"}, ["run_dir=r"])


def test_ranks_share_the_steps_tokens_and_need_the_shadowspill_engine(tmp_path):
    from workloads.recipes.text.experiment.group import rank_options, ranks

    options = {"ranks": 2, "max_tokens_per_step": 4096, "max_seq_len": 512}
    assert ranks(options) == 2 and ranks({}) == 1
    assert rank_options(options)["max_tokens_per_step"] == 2048
    with pytest.raises(ValueError, match="whole sequences for every rank"):
        rank_options({"ranks": 3, "max_tokens_per_step": 4096, "max_seq_len": 512})
    with pytest.raises(ValueError, match="positive integer"):
        ranks({"ranks": 0})
    # a recipe giving a rank an unequal share says its tokens outright
    uneven = {**options, "rank_tokens_per_step": 3072}
    assert rank_options(uneven)["max_tokens_per_step"] == 3072
    with pytest.raises(ValueError, match="whole sequences"):
        rank_options({**options, "rank_tokens_per_step": 3000})
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "engine": "pytorch",
                "ranks": 2,
                "checkpoint": {"family": "llama3", "directory": "/m"},
                "data": "d",
                "max_seq_len": 64,
                "max_tokens_per_step": 128,
                "steps": 1,
                "run_dir": str(tmp_path / "run"),
            }
        )
    )
    with pytest.raises(ValueError, match="ranks needs the shadowspill engine"):
        run_config(path)
    with pytest.raises(ValueError, match="host copies in the profile phase"):
        run_config(path, ["engine=shadowspill", "ranks=1"], phase="communicate")
    with pytest.raises(ValueError, match="carries its own nodes"):
        run_config(path, ["engine=shadowspill", 'heterogeneous={"nodes": []}'])
