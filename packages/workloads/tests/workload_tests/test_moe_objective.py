"""Qwen's nonadditive router term is checked at each actual microbatch geometry."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from shadowspill.training.definitions import load_registry
from shadowspill_client.protocol.operations import OperationKey
from shadowspill_workloads.definitions.moe_objective import Config, bind, compute
from shadowspill_workloads.lora import configure_lora
from workload_tests.test_qwen_moe import (
    Qwen3MoE,
    Qwen3MoEConfig,
    Qwen35MoE,
    Qwen35MoEConfig,
    tiny,
)


def physical(tokens, lengths, *, weighted):
    size = len(tokens)
    inputs = {
        "tokens": torch.tensor([tokens + [0] * (8 - size)]),
        "sequence_lengths": torch.tensor([*lengths, 8 - size], dtype=torch.int32),
    }
    loss_inputs = {
        "target_tokens": (inputs["tokens"] + 1) % 97,
        "loss_mask": torch.arange(8)[None, :] < size,
        "weights": torch.tensor(
            [
                [{1: 1, 2: -0.5, 3: 0, 4: 0.7, 5: 1.2}[t] for t in tokens]
                + [0] * (8 - size)
            ]
        )
        if weighted
        else torch.ones(1, 8),
    }
    return inputs, loss_inputs


def independent(model, inputs, loss_inputs, coefficient):
    """Full projection and explicit router formula, without the objective/head."""
    logits = []
    hooks = [
        block.moe.register_forward_pre_hook(
            lambda module, args: logits.append(
                F.linear(args[0].float(), module.router.weight.float()).reshape(
                    -1, model.config.n_experts
                )
            )
        )
        for block in model.blocks
    ]
    try:
        output = model(**inputs).float()
    finally:
        for hook in hooks:
            hook.remove()
    routes = torch.cat(logits)
    selected = routes.topk(model.config.top_k, dim=-1).indices
    frequency = F.one_hot(selected, model.config.n_experts).float().sum((0, 1))
    frequency = frequency / routes.shape[0]
    auxiliary = model.config.n_experts * (frequency * routes.softmax(-1).mean(0)).sum()
    valid = loss_inputs["loss_mask"]
    targets = torch.where(valid, loss_inputs["target_tokens"], 0)
    ce = F.cross_entropy(output.reshape(-1, 97), targets.flatten(), reduction="none")
    ce = (ce.reshape_as(valid) * loss_inputs["weights"] * valid).sum()
    auxiliary_sum = auxiliary * valid.sum()
    return ce + coefficient * auxiliary_sum, ce, auxiliary_sum


@pytest.mark.parametrize(
    "cls,config",
    [
        (Qwen3MoE, Qwen3MoEConfig()),
        (Qwen35MoE, Qwen35MoEConfig()),
    ],
    ids=["qwen3", "qwen35"],
)
@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.parametrize("split", [False, True])
def test_whole_and_split_each_match_independent_loss_all_gradients_and_two_updates(
    cls, config, lora, split
):
    torch.manual_seed(41)
    model = cls(
        replace(
            tiny(config),
            n_layers=2,
            full_attention_interval=2 if config.attention_gate else 1,
        )
    )
    if lora:
        model = configure_lora(
            model,
            rank=2,
            alpha=2,
            targets=["blocks.*.mixer.w*", "blocks.*.moe"],
            factor_dtype="float32",
        )
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name.endswith(("lora_b", "lora_gate_up_b", "lora_down_b")):
                    parameter.normal_(std=0.02)
    reference = deepcopy(model)
    actual_parameters = [p for p in model.parameters() if p.requires_grad]
    expected_parameters = [p for p in reference.parameters() if p.requires_grad]
    actual_optimizer = torch.optim.SGD(actual_parameters, lr=0.01)
    expected_optimizer = torch.optim.SGD(expected_parameters, lr=0.01)
    batches = (
        [physical([1, 2, 3], [3], weighted=True), physical([4, 5], [2], weighted=True)]
        if split
        else [physical([1, 2, 3, 4, 5], [3, 2], weighted=True)]
    )
    for coefficient in (0.001, 0.07):
        actual_optimizer.zero_grad()
        expected_optimizer.zero_grad()
        for inputs, fields in batches:
            result = compute(
                model,
                inputs,
                fields,
                {"router_aux_loss_coef": torch.tensor(coefficient)},
                tensors=True,
            )
            loss, ce, auxiliary = independent(reference, inputs, fields, coefficient)
            torch.testing.assert_close(result.loss, loss, atol=3e-6, rtol=3e-5)
            torch.testing.assert_close(result.metrics["ce_sum"], ce)
            torch.testing.assert_close(result.metrics["auxiliary_sum"], auxiliary)
            assert result.metrics["valid_tokens"] == fields["loss_mask"].sum()
            assert not result.metrics["logprobs"][~fields["loss_mask"]].any()
            (result.loss * 0.3).backward()
            (loss * 0.3).backward()
        for actual, expected in zip(
            actual_parameters, expected_parameters, strict=True
        ):
            torch.testing.assert_close(actual.grad, expected.grad, atol=2e-6, rtol=5e-4)
        actual_optimizer.step()
        expected_optimizer.step()
        for actual, expected in zip(
            model.parameters(), reference.parameters(), strict=True
        ):
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=5e-4)


@pytest.mark.parametrize(
    "cls,config",
    [
        (Qwen3MoE, Qwen3MoEConfig()),
        (Qwen35MoE, Qwen35MoEConfig()),
    ],
)
def test_default_objective_preserves_existing_loss_and_masked_gradient(cls, config):
    torch.manual_seed(19)
    model = cls(
        replace(
            tiny(config),
            n_layers=2,
            full_attention_interval=2 if config.attention_gate else 1,
        )
    )
    inputs, fields = physical([1, 2, 3, 4, 5], [3, 2], weighted=False)
    result = compute(
        model,
        inputs,
        fields,
        {"router_aux_loss_coef": torch.tensor(model.config.router_aux_loss_coef)},
    )
    legacy = model.loss(
        inputs["tokens"],
        torch.where(fields["loss_mask"], fields["target_tokens"], -100),
        seq_lens=inputs["sequence_lengths"],
        reduction="sum",
    )
    torch.testing.assert_close(result.loss, legacy)
    parameters = tuple(model.parameters())
    torch.testing.assert_close(
        torch.autograd.grad(result.loss, parameters),
        torch.autograd.grad(legacy, parameters),
        atol=2e-6,
        rtol=5e-4,
    )
    fields["loss_mask"].zero_()
    result = compute(
        model, inputs, fields, {"router_aux_loss_coef": torch.tensor(0.07)}
    )
    assert result.loss.item() == 0
    assert all(not g.any() for g in torch.autograd.grad(result.loss, parameters))


def test_declared_objective_is_custom_and_forward_needs_no_auxiliary_controls():
    registry = load_registry(["shadowspill-workloads"])
    model = registry.resolve_model("shadowspill/llama3@1")
    with pytest.raises(ValueError, match="capabilit"):
        registry.resolve_objective("shadowspill/qwen_moe_cross_entropy@1", model=model)
    config = Config(97, "position", "pack", (("tokens",),), 0.001)
    for value in (True, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="router_aux_loss_coef"):
            replace(config, router_aux_loss_coef=value)
    cls = Qwen3MoE(replace(tiny(Qwen3MoEConfig()), n_layers=1))
    inputs, fields = physical([1, 2], [2], weighted=False)
    forward = bind(config, OperationKey("forward", "logprobs", "evaluation"))
    actual = forward(cls, inputs, {"target_tokens": fields["target_tokens"]}, {})
    torch.testing.assert_close(
        actual["logprobs"],
        cls.target_logprobs(**inputs, targets=fields["target_tokens"]).flatten(),
    )
