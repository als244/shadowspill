"""Public MoE setup, packed contracts and bounded initialization on CPU."""

import json
import subprocess
import sys
from dataclasses import asdict, replace

import pytest
import torch

from shadowspill.artifacts.tensors import TensorChunk
from shadowspill.training.definitions import load_registry
from shadowspill.training.definitions.initialization import (
    InitializationContext,
    InitializationRuntime,
)
from shadowspill.training.preparation.layouts import prepare_layouts
from shadowspill.training.preparation.setup import resolve_training
from shadowspill.training.preparation.templates import capture_inventory
from shadowspill_client import (
    BatchConfig,
    BuildCandidates,
    LoRA,
    Memory,
    Model,
    Objective,
    PlanningOptions,
    Preparation,
    Weights,
)
from shadowspill_client.protocol.operations import OperationKey
from shadowspill_workloads.models.olmoe.config import Config as OLMoEConfig
from workload_tests.test_llama_definition import bound, datums
from workload_tests.test_moe_objective import independent
from workload_tests.test_qwen_moe import Qwen3MoEConfig, Qwen35MoEConfig, tiny

FAMILIES = ("qwen3moe", "qwen35moe", "olmoe")


def balancing(family, **overrides):
    config = (
        {"layer_reduction": "sum", "normalize_top_k": True} if family == "olmoe" else {}
    )
    return Objective("shadowspill/moe_balance@1", config={**config, **overrides})


def settings(family, *, dtype="float32", lora=False):
    if family == "olmoe":
        architecture = OLMoEConfig(2, 32, 4, 2, 8, 4, 2, 16, 97, max_seq_len=8)
    else:
        config = Qwen3MoEConfig() if family == "qwen3moe" else Qwen35MoEConfig()
        architecture = replace(
            tiny(config),
            n_layers=2,
            max_seq_len=8,
            full_attention_interval=2 if family == "qwen35moe" else 1,
        )
    projection = "attn" if family == "olmoe" else "mixer"
    return Model(
        f"shadowspill/{family}@1",
        config={"architecture": asdict(architecture)},
        precision={"compute_dtype": dtype, "lora_dtype": "float32"},
        lora=LoRA(2, targets=[f"blocks.*.{projection}.w*", "blocks.*.moe"])
        if lora
        else None,
        initialize=Weights.random(seed=17),
    )


def prepared(family, *, lora=False, dtype="float32", include_padding=False):
    registry = load_registry(["shadowspill-workloads"])
    setup = resolve_training(
        registry,
        model=settings(family, lora=lora, dtype=dtype),
        objective=Objective.sum(
            {
                "tokens": Objective("shadowspill/cross_entropy@1"),
                "balance": balancing(family, include_padding=include_padding),
            },
            weights={"balance": 0.001},
        ),
        batch=BatchConfig(
            limits={"token_capacity": 8, "max_sequence_length": 4, "max_sequences": 4}
        ),
        memory=Memory(1, 1),
        operations={
            "forward_backward": ["metrics", "logprobs"],
            "forward": ["logprobs"],
        },
        prepare=Preparation(
            candidates=BuildCandidates(microbatch_sizes=[8]),
            planning=PlanningOptions(microbatch_counts=[1]),
        ),
    )
    return registry, setup, prepare_layouts(setup)


@pytest.mark.parametrize("family", FAMILIES)
def test_resolved_setup_roundtrip_packing_and_source_pinned_inventory(family):
    registry, setup, layouts = prepared(family)
    assert resolve_training(registry, setup.config).config == setup.config
    inventory = capture_inventory(layouts)
    assert inventory.templates
    for template in inventory.templates.values():
        assert template.model_build.grad_dtype == "float32"
        assert (
            "torch.ops.mlops.lora_router_logits.default"
            in template.model_build.allocation_invariants
        )
    with bound(setup, layouts, datums()) as (binding, adapted):
        inputs, fields, controls = binding.arguments(adapted)
        assert inputs["tokens"].shape == (1, 8)
        assert inputs["sequence_lengths"].sum() == 8
        assert sorted(inputs["sequence_lengths"].tolist()) == [0, 0, 2, 3, 3]
        assert fields["router_mask"].flatten().tolist() == [True] * 5 + [False] * 3
        assert fields["loss_mask"].sum() == 4
        assert controls["balance.weight"].item() == pytest.approx(0.001)
        assert binding.signature()["objective"]["padding_safe"]
    key = OperationKey("forward", "logprobs", "evaluation")
    assert layouts.families[key][0].bindings[0, 0].contract.outputs.tensors.keys() == {
        "tokens.logprobs"
    }


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("lora", [False, True])
def test_bound_objective_matches_independent_full_projection_and_router_gradients(
    family, lora
):
    _, setup, layouts = prepared(family, lora=lora)
    model = setup.model.create()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith(("lora_b", "lora_gate_up_b", "lora_down_b")):
                parameter.normal_(std=0.02)
    with bound(setup, layouts, datums()) as (binding, adapted):
        args = binding.arguments(adapted)
        result = binding.objective(model, *args)
        expected, _, _ = independent(
            model, args[0], args[1], 0.001, include_padding=False
        )
        torch.testing.assert_close(result.loss, expected, atol=3e-6, rtol=3e-5)
        parameters = tuple(p for p in model.parameters() if p.requires_grad)
        torch.testing.assert_close(
            torch.autograd.grad(result.loss, parameters),
            torch.autograd.grad(expected, parameters),
            atol=3e-6,
            rtol=5e-4,
        )


@pytest.mark.parametrize("family", FAMILIES)
def test_bfloat16_construction_preserves_declared_fp32_state_and_restores_default(
    family,
):
    _, setup, _ = prepared(family, dtype="bfloat16")
    previous = torch.get_default_dtype()
    model = setup.model.create()
    assert torch.get_default_dtype() == previous
    assert model.embed.weight.dtype == torch.bfloat16
    assert model.rotary.cosine.dtype == torch.float32
    if family == "qwen35moe":
        assert model.blocks[0].mixer.A_log.dtype == torch.float32
        assert model.blocks[0].mixer.dt_bias.dtype == torch.float32


@pytest.mark.parametrize("family", FAMILIES)
def test_bounded_initialization_is_chunk_invariant_for_full_and_lora_state(
    tmp_path, family
):
    _, setup, _ = prepared(family, lora=True)
    model = setup.model.create()
    spec = setup.model.describe_initialization(
        InitializationContext(
            setup.model.settings,
            setup.model.settings.initialize,
            setup.context.parallelism,
            0,
        )
    )
    with spec.open(
        InitializationRuntime(tmp_path, lambda _: pytest.fail("unexpected artifact"))
    ) as fill:
        for name, tensor in (*model.named_parameters(), *model.named_buffers()):
            count = tensor.numel()
            whole = torch.empty(count, dtype=tensor.dtype)
            pieces = torch.empty_like(whole)
            fill(TensorChunk(name, tuple(tensor.shape), tensor.dtype, 0, count), whole)
            for start in range(0, count, 113):
                stop = min(start + 113, count)
                fill(
                    TensorChunk(name, tuple(tensor.shape), tensor.dtype, start, count),
                    pieces[start:stop],
                )
            torch.testing.assert_close(whole, pieces, rtol=0, atol=0)
            if name.startswith("rotary."):
                torch.testing.assert_close(whole, tensor.flatten(), rtol=0, atol=0)


@pytest.mark.parametrize(
    "configuration",
    [
        {"architecture": {"n_layers": True}},
        {"architecture": {"d_ff_shared": -1}},
        {"architecture": {"router_aux_loss_coef": float("inf")}},
        {"implementation": "other"},
        {"compute_dtype": "float64"},
    ],
)
def test_invalid_structural_settings_fail_before_construction(configuration):
    registry = load_registry(["shadowspill-workloads"])
    with pytest.raises((ValueError, TypeError)):
        registry.resolve_model("shadowspill/qwen35moe@1", config=configuration)


@pytest.mark.parametrize(
    "configuration",
    [
        {"architecture": {"n_layers": True}},
        {"architecture": {"head_dim": 7}},
        {"architecture": {"top_k": 9}},
        {"architecture": {"rope_base": float("nan")}},
        {"architecture": {"initializer_range": 0}},
        {"implementation": "other"},
        {"grad_dtype": "float64"},
        {"lora": {"rank": 2, "dropout": 0.1}},
    ],
)
def test_invalid_olmoe_settings_fail_before_construction(configuration):
    registry = load_registry(["shadowspill-workloads"])
    with pytest.raises((ValueError, TypeError)):
        registry.resolve_model("shadowspill/olmoe@1", config=configuration)


def test_public_moe_description_is_framework_and_implementation_free():
    script = """
import importlib.abc, json, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'triton', 'fla', 'mlops'}:
            # find_spec provider discovery is permitted; imports are checked below.
            return None
        if '.implementations.' in fullname:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from shadowspill.training.definitions import load_registry
from shadowspill_client import Model, Objective
registry = load_registry(['shadowspill-workloads'])
for family in ('qwen3moe', 'qwen35moe', 'olmoe'):
    model = registry.resolve_model('shadowspill/'+family+'@1')
    objective = Objective.sum({
        'tokens': Objective('shadowspill/cross_entropy@1'),
        'balance': Objective('shadowspill/moe_balance@1'),
    }, weights={'balance': .001})
    resolved = registry.resolve_objective(objective.definition,
        model=model, config=objective.config, defaults=objective.defaults)
    assert resolved.definition.padding_safe
    assert not any(n in sys.modules for n in ('torch', 'triton', 'mlops', 'fla'))
print(json.dumps({'families': 3, 'padding_excluded': True}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script], text=True, capture_output=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == {"families": 3, "padding_excluded": True}
