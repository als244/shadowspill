"""What a Megatron-Core case can ask for, what the harness itself refuses
before launch, and what it resolves for a trainer that cannot import the
catalog. What Megatron and Bridge allow together is theirs to decide."""

import argparse
import dataclasses

import pytest

from baselines.common.launch import command, harness_module
from baselines.common.precision import Unsupported
from baselines.megatron_core.settings import (
    IMPLEMENTATION,
    OPTIONS,
    Settings,
    add_arguments,
    aware,
    from_namespace,
)

#: Where Bridge holds an option, or the trainer's environment.
SECTIONS = {"model", "optimizer", "ddp", "dist", "mixed", "comm", "environment"}

BASE = {
    "model": "llama3-tiny",
    "sequence_length": 64,
    "tokens_per_step": 512,
    "tokens_per_microbatch": 128,
}


def case(**changes):
    return Settings.from_dict({**BASE, **changes})


def test_megatron_options_are_settings_by_their_own_names():
    names = {field.name for field in dataclasses.fields(Settings)}
    assert {option.name for option in OPTIONS} <= names
    assert len({option.name for option in OPTIONS}) == len(OPTIONS)
    for option in OPTIONS:
        assert set(option.section.split("+")) <= SECTIONS
        assert option.help and getattr(Settings(), option.name) is None


def test_the_command_line_reproduces_the_case():
    settings = case(
        grad_dtype="bfloat16",
        opt_state_dtype="bfloat16",
        recompute=None,
        recompute_granularity="selective",
        recompute_modules="core_attn,mlp",
        use_distributed_optimizer=True,
        overlap_grad_reduce=False,
        fine_grained_activation_offloading=True,
        offload_modules="qkv_linear,core_attn",
        activation_offload_fraction=0.5,
        cuda_device_max_connections=16,
    )
    settings.validate(1)
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    assert from_namespace(parser.parse_args(settings.arguments())) == settings
    assert harness_module("megatron_core").ENVIRONMENT == "nvidia-training"
    assert command("megatron_core", "py", 1, settings, "out")[11:] == (
        settings.arguments()
    )


def test_a_replica_is_tensor_pipeline_and_context_ranks():
    settings = case(tensor_model_parallel_size=2, sequence_parallel=True)
    geometry = settings.validate(4)
    assert geometry.world_size == 4 and geometry.replicas == 2
    assert geometry.tokens_per_rank_per_step == 256
    assert "replicas" not in case().validate(2).as_dict()
    with pytest.raises(ValueError, match="replicas"):
        case(tensor_model_parallel_size=3).validate(4)


def test_the_row_names_nvidias_implementation_and_counts_ours():
    settings = case()
    assert settings.describe(1)["model"]["implementation"] == IMPLEMENTATION
    assert settings.request().implementation in ("pytorch", "mlops")
    moe = case(model="qwen3moe-tiny")
    assert moe.request().implementation == "mlops"
    hf = moe.resolve(1)["huggingface_config"]
    assert hf["architectures"] == ["Qwen3MoeForCausalLM"]
    with pytest.raises(Unsupported, match="implementation must be"):
        case(implementation="mlops").validate(1)


def test_a_plain_norm_qwen35_has_no_hugging_face_counterpart():
    with pytest.raises(ValueError, match="zero-centers"):
        case(model="qwen35-tiny").resolve(1)
    hf = case(model="qwen35-tiny", model_config={"zero_centered_norm": True})
    assert hf.resolve(1)["huggingface_config"]["model_type"] == "qwen3_5_text"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"recompute": 0.5, "recompute_granularity": "full"}, "alone"),
        ({"recompute": 0.5, "recompute_method": "uniform"}, "recompute_method"),
        ({"recompute": 0.5, "num_layers_in_first_pipeline_stage": 1}, "alone"),
        (
            {
                "recompute_granularity": "full",
                "recompute_method": "uniform",
                "recompute_num_layers": 0,
            },
            "at least 1",
        ),
        ({"optimizer_cuda_graph": True}, "captured at update 3"),
        (
            {
                "use_megatron_fsdp": True,
                "megatron_fsdp_version": 2,
                "grad_dtype": "bfloat16",
            },
            "version 2",
        ),
        (
            {"mode": "lora", "lora": {"rank": 4, "factor_dtype": "float32"}},
            "model's dtype",
        ),
        ({"master_dtype": None}, "only under Megatron-FSDP"),
        (
            {
                "master_dtype": None,
                "opt_state_dtype": "parameter",
                "use_megatron_fsdp": True,
            },
            "float32",
        ),
        (
            {"opt_state_dtype": "bfloat16", "use_distributed_optimizer": False},
            "precision-aware",
        ),
        (
            {"master_dtype": "float16", "use_precision_aware_optimizer": False},
            "turns off",
        ),
        (
            {"model_dtype": "float32", "grad_dtype": "bfloat16"},
            "weights' dtype",
        ),
        ({"gradient_accumulation_fusion": False}, "disagree"),
        ({"verify_reference": True}, "reference check"),
        ({"mode": "lora", "lora": {"rank": 4, "dropout": 0.1}}, "LoRA options"),
        ({"moe_token_dispatcher_type": "deepep"}, "must be one of"),
        ({"recompute_modules": "core_attn,attention"}, "takes"),
        ({"tensor_model_parallel_size": "2"}, "must be an integer"),
        ({"cuda_device_max_connections": 0}, r"\[1, 32\]"),
        ({"cuda_graph_impl": "transformer_engine"}, "captured at update 3"),
        (
            {"cuda_graph_impl": "local", "cuda_graph_warmup_steps": 5},
            "warmup updates",
        ),
    ],
)
def test_the_harness_refuses_what_it_cannot_say_or_measure(changes, message):
    with pytest.raises(ValueError, match=message):
        case(**changes).validate(1)


@pytest.mark.parametrize(
    "changes",
    [
        # Megatron refuses each of these as it sets up, with its message.
        {"cpu_offloading": True, "cpu_offloading_num_layers": 1, "recompute": 0.5},
        {"fine_grained_activation_offloading": True, "offload_modules": "attn_proj"},
        {"tensor_model_parallel_size": 3},
        # Megatron or Bridge accepts each, ignoring or turning off a part.
        {
            "recompute": None,
            "recompute_granularity": "selective",
            "fine_grained_activation_offloading": True,
            "offload_modules": "core_attn",
        },
        {
            "recompute": 0.5,
            "fine_grained_activation_offloading": True,
            "offload_modules": "qkv_linear",
        },
        {"sequence_parallel": True},
        {"tp_comm_overlap": True},
        {"min_offloaded_tensor_size": 1},
        {"optimizer_cpu_offload": True},
    ],
)
def test_megatrons_own_rules_are_left_to_megatron(changes):
    case(**changes).validate(changes.get("tensor_model_parallel_size", 1))


def test_the_precision_aware_optimizer_is_what_the_contract_needs():
    def needs(**fields):
        return aware(case(**fields).precision())

    assert not needs()
    assert not needs(grad_dtype="bfloat16")
    assert needs(master_dtype="float16")
    assert needs(opt_state_dtype="bfloat16")
    assert not needs(opt_state_dtype="parameter")


def test_narrow_gradients_fuse_their_accumulation_as_the_contract_says():
    def fusing(**fields):
        settings = case(**fields)
        settings.validate(1)
        return settings.fusing(settings.precision())

    assert fusing() is True
    assert fusing(fuse_narrow_gradient_accumulation=False) is False
    assert fusing(gradient_accumulation_fusion=True) is True
    assert fusing(grad_dtype="float32") is None
    assert fusing(grad_dtype="float32", gradient_accumulation_fusion=False) is False


def test_the_environment_is_the_cases_own():
    assert case().environment() == {
        "NVTE_CPU_OFFLOAD_V1": None,
        "CUDA_DEVICE_MAX_CONNECTIONS": None,
    }
    offloading = case(
        fine_grained_activation_offloading=True,
        offload_modules="core_attn,attn_proj",
        cuda_device_max_connections=32,
    )
    offloading.validate(1)
    assert offloading.environment() == {
        "NVTE_CPU_OFFLOAD_V1": "1",
        "CUDA_DEVICE_MAX_CONNECTIONS": "32",
    }


def test_selective_recomputation_defaults_to_attention_norms_and_activation():
    def modules(**fields):
        settings = case(**fields)
        settings.validate(1)
        return settings.resolve(1)["recompute_modules"]

    assert modules() is None
    assert modules(recompute_granularity="selective") == ["core_attn", "layernorm"]
    assert modules(recompute_granularity="selective", model="qwen3moe-tiny") == [
        "core_attn",
        "layernorm",
        "moe_act",
    ]
    assert modules(recompute_granularity="selective", recompute_modules="mlp") == [
        "mlp"
    ]


def test_every_option_has_its_flag_in_the_readme():
    from pathlib import Path

    readme = (
        Path(__file__).parents[2] / "baselines/megatron_core/README.md"
    ).read_text()
    missing = [
        option.name
        for option in OPTIONS
        if f"--{option.name.replace('_', '-')}" not in readme
        and f"--[no-]{option.name.replace('_', '-')}" not in readme
    ]
    assert missing == []
