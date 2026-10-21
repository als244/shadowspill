"""What an FSDP2 case can ask for, and what the harness refuses."""

import argparse

import pytest

from baselines.common.launch import command, harness_module
from baselines.common.precision import Precision, Unsupported
from baselines.fsdp2.settings import Settings, add_arguments, from_namespace

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
    )
    settings.validate(2)
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    assert from_namespace(parser.parse_args(settings.arguments())) == settings
    assert harness_module("fsdp2").ENVIRONMENT == "baselines"
    assert command("fsdp2", "py", 2, settings, "out")[11:] == settings.arguments()


def test_gradients_default_to_float32_where_the_shared_default_is_the_model_dtype():
    assert Precision().grad_dtype is None
    assert Settings().grad_dtype == "float32"
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    assert parser.get_default("grad_dtype") == "float32"
    Settings.from_dict(BASE).validate(1)
    with pytest.raises(Unsupported, match="grad_dtype must be float32"):
        Settings.from_dict({**BASE, "grad_dtype": None}).validate(1)


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


def test_a_case_names_its_tokens_per_step():
    with pytest.raises(ValueError, match="tokens_per_step"):
        Settings.from_dict({"model": "llama3-tiny"}).validate(1)


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
        Settings.from_dict({**BASE, **override}).validate(1)


@pytest.mark.parametrize(
    "override",
    [
        {"model_dtype": "float16"},
        {"master_dtype": "bfloat16"},
        {"grad_dtype": "bfloat16"},
        {"opt_state_dtype": "bfloat16"},
        {"opt_state_rounding": "stochastic"},
        {
            "master_dtype": None,
            "grad_dtype": None,
            "opt_state_dtype": "parameter",
            "fuse_narrow_gradient_accumulation": True,
        },
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
        Settings.from_dict({**BASE, **override}).validate(1)


def test_a_pure_bf16_case_steps_the_weights_themselves():
    settings = Settings.from_dict(
        {
            **BASE,
            "master_dtype": None,
            "grad_dtype": None,
            "opt_state_dtype": "parameter",
            "parameter_rounding": "stochastic",
        }
    )
    settings.validate(1)
    assert settings.precision().resolved()["opt_state_dtype"] == "bfloat16"
    low_bit = Settings.from_dict(
        {**BASE, "optimizer": "adamw4bit", "opt_state_dtype": None}
    )
    low_bit.validate(1)
    assert low_bit.describe(1)["precision"]["opt_state_dtype"] is None


def test_a_checkpoint_must_hold_the_requested_model(tmp_path):
    import json

    directory = tmp_path / "checkpoint"
    directory.mkdir()
    tiny = {
        "model_type": "llama",
        "num_hidden_layers": 2,
        "hidden_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 256,
        "vocab_size": 512,
        "rope_theta": 500000.0,
        "tie_word_embeddings": False,
    }
    (directory / "config.json").write_text(json.dumps(tiny))
    (directory / "model.safetensors").write_bytes(
        (8).to_bytes(8, "little") + b"{}      "
    )
    settings = Settings.from_dict({**BASE, "checkpoint": str(directory)})
    settings.validate(1)
    with pytest.raises(ValueError, match="different"):
        Settings.from_dict(
            {**BASE, "model": "llama3-1b", "checkpoint": str(directory)}
        ).validate(1)
    with pytest.raises(ValueError, match="checkpoint directory"):
        Settings.from_dict({**BASE, "checkpoint": str(tmp_path)}).validate(1)
    with pytest.raises(ValueError, match="initialization"):
        Settings.from_dict(
            {**BASE, "checkpoint": str(directory), "verify_reference": True}
        ).validate(1)
