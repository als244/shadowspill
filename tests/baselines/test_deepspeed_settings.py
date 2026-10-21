"""What a DeepSpeed case can ask for, what the harness refuses, and the
configuration it runs under."""

import argparse

import pytest

from baselines.common.geometry import Geometry
from baselines.common.launch import command, harness_module
from baselines.common.precision import Unsupported
from baselines.deepspeed.settings import (
    Settings,
    add_arguments,
    deepspeed_config,
    from_namespace,
)

BASE = {
    "model": "llama3-tiny",
    "tokens_per_step": 1024,
    "tokens_per_microbatch": 128,
    "sequence_length": 64,
    "grad_dtype": "bfloat16",
}


def test_flags_round_trip_every_setting():
    settings = Settings.from_dict(
        {
            **BASE,
            "zero_stage": 3,
            "offload_param": "cpu",
            "offload_optimizer": "cpu",
            "activation_offload": "cpu",
            "contiguous_checkpointing": True,
            "grad_dtype": "bfloat16",
        }
    )
    settings.validate(2)
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    assert from_namespace(parser.parse_args(settings.arguments())) == settings
    assert harness_module("deepspeed").ENVIRONMENT == "baselines"
    assert command("deepspeed", "py", 2, settings, "out")[11:] == settings.arguments()
    assert "zero_stage" in settings.describe(2)["configuration"]


def test_the_configuration_follows_the_settings():
    settings = Settings.from_dict({**BASE, "zero_stage": 3, "offload_optimizer": "cpu"})
    geometry = Geometry(64, 1024, 128, 2)
    config = deepspeed_config(settings, geometry, settings.precision())
    assert "communication_data_type" not in config and "data_types" not in config
    stage_one = Settings.from_dict({**BASE, "zero_stage": 1, "grad_dtype": "float32"})
    full = deepspeed_config(stage_one, geometry, stage_one.precision())
    assert full["communication_data_type"] == "fp32"
    assert full["data_types"] == {"grad_accum_dtype": "fp32"}
    assert config["train_micro_batch_size_per_gpu"] == 2
    assert config["gradient_accumulation_steps"] == 4
    assert config["train_batch_size"] == 16
    assert config["zero_optimization"]["stage"] == 3
    assert config["zero_optimization"]["offload_optimizer"]["device"] == "cpu"
    assert "offload_param" not in config["zero_optimization"]
    assert config["bf16"] == {"enabled": True}
    half = Settings.from_dict(
        {**BASE, "model_dtype": "float16", "grad_dtype": "float16"}
    )
    config = deepspeed_config(half, geometry, half.precision())
    assert config["fp16"]["enabled"] and config["fp16"]["loss_scale"] == 0
    assert "compile" not in config
    compiled = Settings.from_dict(
        {
            **BASE,
            "zero_stage": 3,
            "recompute": 0,
            "deepcompile": True,
            "deepcompile_offload": "activations",
            "warmup_steps": 8,
        }
    )
    compiled.validate(1)
    config = deepspeed_config(compiled, geometry, compiled.precision())
    assert config["compile"] == {
        "deepcompile": True,
        "offload_activation": True,
        "offload_opt_states": False,
        "offload_parameters": False,
        "free_activation": False,
    }


@pytest.mark.parametrize(
    "override",
    [
        {"zero_stage": 0},
        {"offload_optimizer": "disk"},
        {"offload_optimizer": "nvme"},
        {"activation_offload": "cpu", "recompute": 0},
        {"reduce_bucket_size": 0},
        {"verify_reference": True, "model_dtype": "float16", "grad_dtype": "float16"},
        {"deepcompile_offload": "activations"},
        {"deepcompile_free_activation": True},
        {"deepcompile": True, "warmup_steps": 7},
        {"deepcompile": True, "deepcompile_backend": "aot"},
    ],
)
def test_invalid_settings_are_refused(override):
    with pytest.raises(ValueError):
        Settings.from_dict({**BASE, **override}).validate(1)


@pytest.mark.parametrize(
    "override",
    [
        {"zero_stage": 1, "offload_optimizer": "cpu"},
        {"zero_stage": 2, "grad_dtype": "float32"},
        {"zero_stage": 3, "grad_dtype": "float32"},
        {"zero_stage": 2, "offload_param": "cpu"},
        {"opt_state_dtype": "bfloat16"},
        {"master_dtype": None, "opt_state_dtype": "float32"},
        {"master_dtype": None, "opt_state_dtype": None},
        {
            "master_dtype": None,
            "opt_state_dtype": "parameter",
            "zero_stage": 1,
            "grad_dtype": "float32",
        },
        {
            "master_dtype": None,
            "opt_state_dtype": "parameter",
            "model_dtype": "float16",
            "grad_dtype": "float16",
        },
        {
            "master_dtype": None,
            "opt_state_dtype": "parameter",
            "deepcompile": True,
            "warmup_steps": 8,
            "recompute": 0,
            "zero_stage": 3,
        },
        {"grad_dtype": "float16"},
        {"parameter_rounding": "stochastic"},
        {"fuse_narrow_gradient_accumulation": True},
        {"deepcompile": True, "warmup_steps": 8, "deepcompile_offload": "activations"},
        {
            "deepcompile": True,
            "warmup_steps": 8,
            "zero_stage": 3,
            "deepcompile_offload": "optimizer-states",
            "offload_optimizer": "cpu",
        },
        {
            "deepcompile": True,
            "warmup_steps": 8,
            "zero_stage": 3,
            "offload_optimizer": "cpu",
            "offload_param": "cpu",
        },
        {"deepcompile": True, "warmup_steps": 8, "deepcompile_free_activation": True},
        {"deepcompile": True, "warmup_steps": 8, "activation_offload": "cpu"},
        {"deepcompile": True, "warmup_steps": 8, "recompute": 0.5},
        {"deepcompile": True, "warmup_steps": 8, "recompute": 0, "zero_stage": 1},
        {
            "deepcompile": True,
            "warmup_steps": 8,
            "recompute": 0,
            "zero_stage": 1,
            "tokens_per_microbatch": 1024,
            "grad_dtype": "float32",
        },
        {
            "deepcompile": True,
            "warmup_steps": 8,
            "recompute": 0,
            "zero_stage": 2,
            "mode": "lora",
            "lora": {"rank": 4},
        },
        {
            "deepcompile": True,
            "warmup_steps": 8,
            "recompute": 0,
            "zero_stage": 3,
            "model_dtype": "float16",
            "grad_dtype": "float16",
        },
        {
            "deepcompile": True,
            "warmup_steps": 8,
            "zero_stage": 3,
            "recompute": 0,
            "deepcompile_offload": "parameters",
        },
    ],
)
def test_what_deepspeed_cannot_honor_is_refused(override):
    with pytest.raises(Unsupported):
        Settings.from_dict({**BASE, **override}).validate(1)


@pytest.mark.parametrize("moments", ["bfloat16", "parameter"])
@pytest.mark.parametrize(
    "zero", [{"zero_stage": 2}, {"zero_stage": 3, "offload_optimizer": "cpu"}]
)
def test_a_bfloat16_model_without_masters_steps_bfloat16_values(zero, moments):
    settings = Settings.from_dict(
        {**BASE, **zero, "master_dtype": None, "opt_state_dtype": moments}
    )
    settings.validate(1)
    assert settings.precision().resolved()["opt_state_dtype"] == "bfloat16"
    config = deepspeed_config(
        settings, Geometry(64, 1024, 128, 1), settings.precision()
    )
    assert config["bf16"] == {
        "enabled": True,
        "bf16_master_weights_and_grads": True,
        "bf16_optimizer_states": True,
    }


def test_deepcompile_takes_zero_one_without_accumulation():
    Settings.from_dict(
        {
            **BASE,
            "deepcompile": True,
            "warmup_steps": 8,
            "recompute": 0,
            "zero_stage": 1,
            "tokens_per_microbatch": 1024,
        }
    ).validate(1)


def test_deepcompile_offloads_parameters_of_the_pytorch_implementation():
    Settings.from_dict(
        {
            **BASE,
            "deepcompile": True,
            "warmup_steps": 8,
            "zero_stage": 3,
            "recompute": 0,
            "implementation": "pytorch",
            "deepcompile_offload": "parameters",
        }
    ).validate(1)


def test_a_float32_model_steps_its_weights():
    settings = Settings.from_dict(
        {
            **BASE,
            "model_dtype": "float32",
            "master_dtype": None,
            "grad_dtype": None,
            "opt_state_dtype": "parameter",
        }
    )
    settings.validate(1)
    config = deepspeed_config(
        settings, Geometry(64, 1024, 128, 1), settings.precision()
    )
    assert "bf16" not in config and "fp16" not in config
