"""What an FSDP2 case can ask for, and what the harness refuses."""

import argparse

import pytest

from baselines.common.precision import Unsupported
from baselines.fsdp2.settings import Settings, add_arguments, command, from_namespace

BASE = {
    "model": "llama3-tiny",
    "tokens_per_step": 1024,
    "tokens_per_microbatch": 128,
    "sequence_length": 64,
}


def test_flags_round_trip_every_setting():
    settings = Settings.from_dict(
        {
            **BASE,
            "mode": "lora",
            "lora": {"rank": 4},
            "model_config": {"n_layers": 1},
            "master_dtype": None,
            "grad_dtype": None,
            "opt_state_dtype": "parameter",
            "parameter_rounding": "stochastic",
            "activation_offload": "overlapped",
            "backward_prefetch": 2,
            "allocator_cap_gib": 20.5,
        }
    ).normalized(2)
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    assert from_namespace(parser.parse_args(settings.arguments())) == settings
    line = command("py", 2, settings, "out")
    assert line[:4] == ["py", "-m", "torch.distributed.run", "--standalone"]
    assert line[7:10] == ["baselines.fsdp2.train", "--output", "out"]


def test_describe_splits_model_geometry_precision_and_configuration():
    description = Settings.from_dict(BASE).describe(2)
    assert description["model"] == {
        "model": "llama3-tiny",
        "implementation": "mlops",
        "mode": "full",
        "lora": {},
        "config": {},
    }
    assert description["geometry"]["accumulation"] == 4
    assert description["precision"]["master_dtype"] == "float32"
    configuration = description["configuration"]
    assert "recompute" in configuration and "data" in configuration
    assert "tokens_per_step" not in configuration
    assert "model_dtype" not in configuration


def test_a_tier_resolves_to_its_tokens_per_step():
    settings = Settings.from_dict(
        {"model": "llama3-tiny", "tier": "rtx5090", "tokens_per_microbatch": 8192}
    ).normalized(1)
    assert settings.tokens_per_step == 131072 and settings.tier is None
    with pytest.raises(ValueError):
        Settings.from_dict(
            {"model": "llama3-tiny", "tier": "rtx5090", "tokens_per_step": 1}
        ).normalized(1)
    with pytest.raises(ValueError):
        Settings.from_dict({"model": "llama3-tiny"}).normalized(1)


@pytest.mark.parametrize(
    "override",
    [
        {"model": None},
        {"model": "llama2-7b"},
        {"implementation": "jax"},
        {"lora": {"rank": 4}},
        {"lora": "rank=4"},
        {"tokens_per_microbatch": 100},
        {"recompute": 2},
        {"activation_offload": "disk"},
        {"offload": "nvme"},
        {"backward_prefetch": 0},
        {"optimizer": "sgd"},
        {"qlora_nf4": True},
        {"mode": "lora", "float8": True, "qlora_nf4": True},
        {"data": "/nonexistent/corpus"},
        {"steps": 0},
        {"verify_reference": True, "model": "llama3-1b"},
        {"verify_reference": True, "optimizer": "adamw8bit", "opt_state_dtype": None},
        {"pin_memory": "yes"},
        {"prefetch_typo": 1},
    ],
)
def test_invalid_settings_are_refused(override):
    with pytest.raises(ValueError):
        Settings.from_dict({**BASE, **override}).normalized(1)


@pytest.mark.parametrize(
    "override",
    [
        {"model_dtype": "float16"},
        {"master_dtype": "bfloat16"},
        {"grad_dtype": "bfloat16"},
        {"opt_state_dtype": "bfloat16"},
        {"opt_state_rounding": "stochastic"},
        {"round_accumulation_once": True},
        {"parameter_rounding": "stochastic"},
        {"optimizer": "adamw8bit"},
        {"float8": True, "mode": "lora", "lora": {"rank": 4}},
        {
            "float8": True,
            "model_dtype": "float32",
            "master_dtype": None,
            "grad_dtype": None,
            "opt_state_dtype": None,
        },
    ],
)
def test_precision_outside_the_harness_is_refused(override):
    with pytest.raises(Unsupported):
        Settings.from_dict({**BASE, **override}).normalized(1)


def test_a_pure_bf16_case_steps_the_weights_themselves():
    settings = Settings.from_dict(
        {
            **BASE,
            "master_dtype": None,
            "grad_dtype": None,
            "opt_state_dtype": "parameter",
            "parameter_rounding": "stochastic",
        }
    ).normalized(1)
    assert settings.precision().resolved()["opt_state_dtype"] == "bfloat16"
    low_bit = Settings.from_dict(
        {**BASE, "optimizer": "adamw4bit", "opt_state_dtype": None}
    ).normalized(1)
    assert low_bit.describe(1)["precision"]["opt_state_dtype"] is None
