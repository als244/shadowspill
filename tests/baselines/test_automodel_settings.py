"""What an AutoModel case can ask for, what the harness itself refuses
before launch, and how the precision contract maps onto AutoModel's
options. What AutoModel allows together is its own to decide."""

import argparse
import dataclasses
import json

import pytest

from baselines.automodel.settings import (
    IMPLEMENTATION,
    OPTIONS,
    Settings,
    add_arguments,
    from_namespace,
)
from baselines.common.launch import command, harness_module

SECTIONS = {
    "distributed",
    "distributed.moe",
    "distributed.pipeline",
    "model",
    "model.backend",
    "model.backend.mok",
    "loss_fn",
    "fp8",
    "compile",
    "step_scheduler",
    "peft",
    "harness",
}

BASE = {
    "model": "llama3-tiny",
    "sequence_length": 64,
    "tokens_per_step": 512,
    "tokens_per_microbatch": 128,
}


def case(**changes):
    return Settings.from_dict({**BASE, **changes})


def test_automodel_options_are_settings_by_their_own_names():
    names = {field.name for field in dataclasses.fields(Settings)}
    assert {option.name for option in OPTIONS} <= names
    assert len({option.name for option in OPTIONS}) == len(OPTIONS)
    for option in OPTIONS:
        assert option.section in SECTIONS
        assert option.help and getattr(Settings(), option.name) is None


def test_the_command_line_reproduces_the_case():
    settings = case(
        loss="chunked",
        chunk_len=256,
        activation_checkpointing="selective",
        attn="te",
        sdpa_method="flash_attention,efficient_attention",
        defer_fsdp_grad_sync=False,
    )
    settings.validate(1)
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    assert from_namespace(parser.parse_args(settings.arguments())) == settings
    assert harness_module("automodel").ENVIRONMENT == "nvidia-training"
    assert command("automodel", "py", 1, settings, "out")[11:] == settings.arguments()
    assert settings.describe(1)["model"]["implementation"] == IMPLEMENTATION


def mechanism(world=1, **fields):
    settings = case(**fields)
    settings.validate(world)
    return settings.mechanism(world)


def test_the_contract_picks_automodels_option():
    default = mechanism()
    assert default["optimizer"] == "fused_adam" and default["master"] == "float32"
    assert default["storage"] == "bfloat16" and default["moments"] == "float32"
    in_place = mechanism(master_dtype=None, opt_state_dtype="parameter")
    assert in_place["optimizer"] == "adamw" and in_place["master"] is None
    assert in_place["moments"] == "bfloat16"
    sharded = mechanism(2, grad_dtype="float32")
    assert sharded["storage"] == "float32" and sharded["mp_param_dtype"] == "bfloat16"
    assert sharded["mp_reduce_dtype"] == "float32" and sharded["optimizer"] == "adamw"
    assert mechanism(opt_state_dtype="bfloat16")["moments"] == "bfloat16"
    assert mechanism(master_dtype="float16")["master"] == "float16"
    full = mechanism(model_dtype="float32", master_dtype=None)
    assert full["storage"] == "float32" and full["optimizer"] == "adamw"
    megatron = mechanism(2, strategy="megatron_fsdp", grad_dtype="float32")
    assert megatron["preserve_fp32_weights"] and megatron["grad_reduce_in_fp32"]
    assert megatron["optimizer"] == "adamw" and megatron["moments"] == "float32"
    assert mechanism(strategy="ddp")["optimizer"] == "fused_adam"
    resolved = case().resolve(1)
    assert resolved["precision_mechanism"] == default
    assert resolved["huggingface_config"]["architectures"] == ["LlamaForCausalLM"]


def test_the_loss_and_its_linear_layers_default_by_tensor_parallelism():
    def chosen(world=1, **fields):
        settings = case(**fields)
        settings.validate(world)
        resolved = settings.resolve(world)
        return resolved["loss"], resolved["linear"]

    assert chosen() == ("chunked", "torch")
    assert chosen(2, tp_size=2) == ("te_parallel", None)
    assert chosen(loss="masked") == ("masked", None)
    assert chosen(linear="te", loss="fused_linear") == ("fused_linear", "te")
    # AutoModel refuses a Transformer Engine head under the chunked loss.
    assert chosen(linear="te") == ("chunked", "te")


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"grad_dtype": "float32"}, "data-parallel ranks"),
        ({"optimizer": "adamw"}, "keeps no master"),
        ({"master_dtype": None, "optimizer": "adam"}, "moments at the stepped"),
        ({"optimizer": "flash_adamw", "master_dtype": None}, "24-bit"),
        ({"mp_reduce_dtype": "float32"}, "disagrees"),
        ({"strategy": "ddp", "model_dtype": "float32", "master_dtype": None}, "casts"),
        ({"recompute": 0.5}, "0 or 1"),
        ({"recompute": 1, "activation_checkpointing": "full"}, "set one"),
        ({"fuse_narrow_gradient_accumulation": True}, "adds it after"),
        ({"verify_reference": True}, "reference check"),
        ({"implementation": "mlops"}, "implementation must be"),
        ({"strategy": "deepspeed"}, "must be one of"),
        ({"sdpa_method": "flash_attention,fast"}, "takes"),
        ({"tp_size": "2"}, "must be an integer"),
        ({"mode": "lora", "lora": {"rank": 4, "dropout": 0.1}}, "LoRA options"),
        (
            {"mode": "lora", "lora": {"rank": 4, "factor_dtype": "float32"}},
            "casts each LoRA module",
        ),
        ({"model_dtype": "float32", "master_dtype": "float16"}, "own float32"),
        ({"model_dtype": "float32", "grad_dtype": "bfloat16"}, "gradients at"),
        ({"cpu_offload": True}, "FusedAdam"),
        ({"strategy": "ddp", "cpu_offload": True}, "are FSDP2's"),
        ({"strategy": "ddp", "mp_output_dtype": "float32"}, "are FSDP2's"),
    ],
)
def test_the_harness_refuses_what_it_cannot_say(changes, message):
    with pytest.raises(ValueError, match=message):
        case(**changes).validate(1)


def test_megatron_fsdp_keeps_float32_main_weights_or_none():
    with pytest.raises(ValueError, match="float32 main weights or none"):
        case(strategy="megatron_fsdp", master_dtype="float16").validate(2)
    with pytest.raises(ValueError, match="disagrees"):
        case(strategy="megatron_fsdp", preserve_fp32_weights=False).validate(2)
    # Its moments follow the masters: float32 with them, bfloat16 without.
    with pytest.raises(ValueError, match="moments at float32"):
        case(strategy="megatron_fsdp", opt_state_dtype="bfloat16").validate(2)
    with pytest.raises(ValueError, match="moments at bfloat16"):
        case(strategy="megatron_fsdp", master_dtype=None).validate(2)
    # Tensor parallelism over every rank leaves nothing to shard.
    with pytest.raises(ValueError, match="data-parallel ranks"):
        case(strategy="megatron_fsdp", grad_dtype="float32", tp_size=2).validate(2)


def test_fsdp2_float32_storage_is_the_float32_master():
    with pytest.raises(ValueError, match="float32 storage"):
        case(grad_dtype="float32", master_dtype="float16").validate(2)
    case(grad_dtype="float32").validate(2)


def test_a_checkpoint_around_a_text_model_is_refused(tmp_path):
    # As the published Qwen 3.5 checkpoints are: the text model's
    # configuration under text_config, beside a vision tower's.
    qwen35 = {"model": "qwen35-tiny", "model_config": {"zero_centered_norm": True}}
    text = case(**qwen35).resolve(1)["huggingface_config"]
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
                "text_config": text,
                "vision_config": {},
            }
        )
    )
    with pytest.raises(ValueError, match="around a text model"):
        case(**qwen35, checkpoint=str(tmp_path)).validate(1)


@pytest.mark.parametrize(
    "changes",
    [
        # AutoModel refuses or ignores each of these as it sets up.
        {"strategy": "ddp", "tp_size": 2},
        {"activation_checkpointing": "selective", "strategy": "megatron_fsdp"},
        {"compile": True},
        {"loss": "chunked", "linear": "te"},
        # Torch's AdamW steps where the shards are.
        {"cpu_offload": True, "master_dtype": None, "opt_state_dtype": "bfloat16"},
    ],
)
def test_automodels_own_rules_are_left_to_automodel(changes):
    case(**changes).validate(changes.get("tp_size", 1))


def test_the_resolved_case_says_whether_the_model_routes_to_experts():
    assert case().resolve(1)["mixture"] is False
    assert case(model="qwen3moe-tiny").resolve(1)["mixture"] is True
    # Factors at another dtype are honored where FSDP2 shards.
    case(mode="lora", lora={"rank": 4, "factor_dtype": "float32"}).validate(2)


def test_every_option_has_its_flag_in_the_readme():
    from pathlib import Path

    readme = (Path(__file__).parents[2] / "baselines/automodel/README.md").read_text()
    missing = [
        option.name
        for option in OPTIONS
        if f"--{option.name.replace('_', '-')}" not in readme
        and f"--[no-]{option.name.replace('_', '-')}" not in readme
    ]
    assert missing == []
