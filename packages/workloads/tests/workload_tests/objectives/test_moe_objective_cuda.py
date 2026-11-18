"""Real provider captures of composed MoE loss and whole/split update oracles."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from engine_tests.precision import bf16_step
from shadowspill.pytorch.capture.objective import capture_training_objective
from shadowspill_client.protocol.operations import OperationKey
from shadowspill_workloads.lora import configure_lora
from workload_tests.objectives.test_moe_objective import (
    OLMoE,
    OLMoEConfig,
    Qwen3MoE,
    Qwen3MoEConfig,
    Qwen35MoE,
    Qwen35MoEConfig,
    independent,
    physical,
    resolve,
    tiny,
)

pytestmark = pytest.mark.cuda


def create(family, lora):
    if family == "olmoe":
        cls, config = OLMoE, OLMoEConfig(2, 64, 2, 2, 32, 4, 2, 64, 97, max_seq_len=64)
    else:
        cls, config = (
            (Qwen3MoE, Qwen3MoEConfig())
            if family == "qwen3"
            else (Qwen35MoE, Qwen35MoEConfig())
        )
        config = replace(
            tiny(config),
            n_layers=2,
            d_model=64,
            n_heads=2,
            n_kv_heads=2,
            head_dim=32,
            d_ff_expert=64,
            max_seq_len=64,
            full_attention_interval=2 if family == "qwen35" else 1,
            lin_k_head_dim=32,
            lin_v_head_dim=32,
        )
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("cuda"):
            model = cls(config)
    finally:
        torch.set_default_dtype(previous)
    if lora:
        model = configure_lora(
            model,
            rank=2,
            alpha=2,
            targets=[
                "blocks.*.attn.w*" if family == "olmoe" else "blocks.*.mixer.w*",
                "blocks.*.moe",
            ],
            factor_dtype="float32",
        )
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name.endswith(("lora_b", "lora_gate_up_b", "lora_down_b")):
                    parameter.normal_(std=0.02)
    return model


@pytest.mark.parametrize("family", ["qwen3", "qwen35", "olmoe"])
@pytest.mark.parametrize("include_padding", [False, True])
@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("lora", [False, True])
def test_provider_capture_and_two_updates_match_independent_geometry_oracle(
    family, include_padding, split, lora
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for provider capture")
    torch.manual_seed(17)
    model = create(family, lora)
    reference = deepcopy(model)
    _, _, objective = resolve(model, include_padding=include_padding)
    controls = {
        name: torch.tensor(value, device="cuda")
        for name, value in objective.defaults.items()
    }
    parts = [([1, 2, 3], [3]), ([4, 5], [2])] if split else [([1, 2, 3, 4, 5], [3, 2])]
    batches = []
    for tokens, lengths in parts:
        inputs, fields = physical(tokens, lengths, weighted=True, size=64)
        batches.append(
            (
                {k: v.cuda() for k, v in inputs.items()},
                {k: v.cuda() for k, v in fields.items()},
            )
        )
    key = OperationKey("forward_backward" if split else "step", "metrics")
    captured = capture_training_objective(
        model, objective.computation(key), [*batches[0], controls]
    )
    graph = captured.exported.exported_program.module()
    for coefficient in (0.001, 0.07):
        controls["balance.weight"] = torch.tensor(coefficient, device="cuda")
        model.zero_grad()
        reference.zero_grad()
        for inputs, fields in batches:
            actual, *_ = graph(inputs, fields, controls)
            expected, _, _ = independent(
                reference, inputs, fields, coefficient, include_padding=include_padding
            )
            torch.testing.assert_close(actual, expected, atol=5e-4, rtol=2e-4)
            (actual * 0.3).backward()
            (expected * 0.3).backward()
        for (name, actual), (_, expected) in zip(
            model.named_parameters(), reference.named_parameters(), strict=True
        ):
            if not actual.requires_grad:
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                continue
            # The two implementations round in a different order.
            torch.testing.assert_close(
                actual.grad,
                expected.grad,
                atol=max(0.002, 2 * bf16_step(expected.grad)),
                rtol=0.03,
                msg=lambda message, name=name: name + "\n" + message,
            )
            with torch.no_grad():
                actual.add_(actual.grad, alpha=-0.01)
                expected.add_(expected.grad, alpha=-0.01)
            torch.testing.assert_close(
                actual,
                expected,
                atol=0.0005,
                rtol=0.02,
                msg=lambda message, name=name: name + "\n" + message,
            )
        # Check each next update from identical weights. Independently rounded
        # BF16 updates can change later routing/activations near a rounding edge;
        # FP32 CPU tests separately check independent two-update trajectories.
        reference.load_state_dict(model.state_dict())
