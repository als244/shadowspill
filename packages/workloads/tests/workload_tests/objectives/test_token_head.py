"""Bounded projections preserve target logprobs and arbitrary token gradients."""

import copy

import pytest
import torch
from torch.nn import functional as F

from shadowspill.pytorch.capture.objective import capture_training_objective
from shadowspill_workloads.models.llama3.config import Config
from shadowspill_workloads.models.llama3.implementations.mlops import (
    Model as MlopsLlama,
)
from shadowspill_workloads.models.llama3.implementations.pytorch import Model as Llama
from shadowspill_workloads.models.shared import token_head


@pytest.mark.parametrize("policy", [False, True])
@pytest.mark.parametrize("frozen_head", [False, True])
def test_installed_token_objective_captures_one_fused_head_and_matches_reference(
    policy, frozen_head
):
    from mlops.objectives import clipped_policy, cross_entropy

    from shadowspill_workloads.components.token import compute
    from shadowspill_workloads.components.token_evaluation import compute as evaluate

    torch.manual_seed(211)
    model = MlopsLlama(Config(1, 16, 2, 1, 32, 31, max_seq_len=8))
    model.lm_head.weight.requires_grad_(not frozen_head)
    tokens = torch.randint(0, 31, (1, 7))
    fields = {
        "target_tokens": torch.randint(0, 31, tokens.shape),
        "response_mask" if policy else "loss_mask": torch.tensor(
            [[True, True, False, True, True, True, True]]
        ),
        "weights": torch.tensor([[1.0, -0.3, float("nan"), 0.7, -1.0, 0.0, 0.2]]),
    }
    controls = {}
    if policy:
        fields.update(
            old_logprobs=torch.full(tokens.shape, -3.2),
            advantages=torch.tensor([[1.0, -1.0, float("nan"), 0.5, -0.5, 0.0, 0.3]]),
            reference_logprobs=torch.full(tokens.shape, -3.1),
        )
        controls = {
            "clip_ratio": torch.tensor(0.2),
            "reference_beta": torch.tensor(0.1),
        }
    mask_name = "response_mask" if policy else "loss_mask"

    def objective(model, inputs, fields, controls):
        features = evaluate(
            model,
            inputs,
            fields,
            {"logprobs": True, "target_mask." + mask_name: True},
        )
        return compute(features, inputs, fields, controls, policy=policy, tensors=True)

    arguments = [{"tokens": tokens}, fields, controls]
    captured = capture_training_objective(model, objective, arguments)
    calls = [
        node
        for node in captured.exported.exported_program.graph.nodes
        if node.target == torch.ops.mlops.head_objective_builtin_chunked_fwd.default
    ]
    assert len(calls) == 1
    assert calls[0].args[6] == ("clipped_policy" if policy else "cross_entropy")
    assert calls[0].args[-1][1] == (not frozen_head)
    result = captured.exported.exported_program.module()(*arguments)
    logprobs = model.target_logprobs(tokens, fields["target_tokens"])
    if policy:
        expected = clipped_policy(
            logprobs,
            fields[mask_name],
            fields["weights"],
            fields["old_logprobs"],
            fields["advantages"],
            controls["clip_ratio"],
            reference_logprobs=fields["reference_logprobs"],
            reference_beta=controls["reference_beta"],
        )
    else:
        expected = cross_entropy(logprobs, fields[mask_name], fields["weights"])
    torch.testing.assert_close(result[0], expected)
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    for actual, reference in zip(
        torch.autograd.grad(result[0], parameters),
        torch.autograd.grad(expected, parameters),
        strict=True,
    ):
        torch.testing.assert_close(actual, reference, atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("chunk", [None, 1, 3, 8, 100])
def test_head_matches_full_projection_for_values_and_token_weighted_vjps(chunk):
    torch.manual_seed(204)
    head = torch.nn.Linear(5, 11)
    hidden = torch.randn(2, 4, 5, requires_grad=True)
    targets = torch.randint(0, 11, (2, 4))
    weights = torch.tensor([[1, -0.3, 0, 0.7], [-1, 0.6, 0.2, 2]])
    actual = token_head.target_logprobs(hidden, head, targets, chunk_size=chunk)
    expected = head(hidden).log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1)
    torch.testing.assert_close(actual, expected)
    parameters = (hidden, *head.parameters())
    actual_grad = torch.autograd.grad((actual * weights).sum(), parameters)
    expected_grad = torch.autograd.grad((expected * weights).sum(), parameters)
    for a, b in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)


def test_default_head_bound_is_independent_of_total_token_count(monkeypatch):
    monkeypatch.setattr(token_head, "TARGET_LOGIT_BYTES", 4 * 11 * 3)

    class Head(torch.nn.Linear):
        def __init__(self):
            super().__init__(5, 11)
            self.rows = []

        def forward(self, hidden):
            self.rows.append(hidden.shape[0])
            return super().forward(hidden)

    head = Head()
    token_head.target_logprobs(
        torch.zeros(37, 5), head, torch.zeros(37, dtype=torch.int64)
    )
    assert max(head.rows) == 3 and sum(head.rows) == 37
    with pytest.raises(ValueError, match="one target per"):
        token_head.target_logprobs(
            torch.zeros(3, 5), head, torch.zeros(2, dtype=torch.int64)
        )
    for bad in (0, -1, 0.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            token_head.target_logprobs(
                torch.zeros(3, 5),
                head,
                torch.zeros(3, dtype=torch.int64),
                chunk_size=bad,
            )


@pytest.mark.parametrize("family", ["qwen3moe", "qwen35moe"])
@pytest.mark.parametrize("lora", [False, True])
def test_qwen_target_logprobs_match_full_head_vjps_and_updates(family, lora):
    from importlib import import_module

    from shadowspill_workloads.lora import configure_lora
    from workload_tests.models.test_qwen_moe import tiny

    torch.manual_seed(208)
    config = import_module(f"shadowspill_workloads.models.{family}.config").Config()
    model_type = import_module(
        f"shadowspill_workloads.models.{family}.implementations.mlops"
    ).Model
    model = model_type(tiny(config))
    if lora:
        configure_lora(model, rank=2, alpha=2, head="lora", factor_dtype="float32")
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_" in name and name.endswith("b"):
                    parameter.normal_(std=0.02)
    oracle = copy.deepcopy(model)
    selected = tuple(p for p in model.parameters() if p.requires_grad)
    reference = tuple(p for p in oracle.parameters() if p.requires_grad)
    optimizer, other = (torch.optim.SGD(p, lr=0.01) for p in (selected, reference))
    tokens = torch.randint(0, model.config.vocab_size, (1, 7))
    targets = torch.randint(0, model.config.vocab_size, tokens.shape)
    weights = torch.tensor([[1, -0.3, 0, 0.7, -1, 0.4, 0.2]])
    for _ in range(2):
        optimizer.zero_grad()
        other.zero_grad()
        actual = model.target_logprobs(tokens, targets, (3, 4), head_chunk_size=2)
        logits = oracle(tokens, (3, 4))
        expected = logits.log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=3e-5)
        (actual * weights).sum().backward()
        (expected * weights).sum().backward()
        for a, b in zip(selected, reference, strict=True):
            torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=3e-5)
        optimizer.step()
        other.step()
        torch.testing.assert_close(
            model.state_dict(), oracle.state_dict(), atol=2e-6, rtol=3e-5
        )


@pytest.mark.parametrize("model_type", [Llama, MlopsLlama])
@pytest.mark.parametrize("mode", ["full", "frozen_head", "lora_head", "tied"])
def test_llama_target_logprobs_match_full_head_gradients_and_updates(model_type, mode):
    from shadowspill_workloads.lora import configure_lora

    torch.manual_seed(205)
    config = Config(1, 16, 2, 1, 32, 31, max_seq_len=16, tied_embeddings=mode == "tied")
    model = model_type(config)
    if mode in ("frozen_head", "lora_head"):
        configure_lora(
            model,
            rank=2,
            alpha=2,
            head="lora" if mode == "lora_head" else "frozen",
            factor_dtype="float32",
        )
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_" in name and name.endswith("b"):
                    parameter.normal_(std=0.02)
    oracle = copy.deepcopy(model)
    selected = tuple(p for p in model.parameters() if p.requires_grad)
    reference = tuple(p for p in oracle.parameters() if p.requires_grad)
    optimizer = torch.optim.SGD(selected, lr=0.01)
    other = torch.optim.SGD(reference, lr=0.01)
    tokens = torch.randint(0, 31, (1, 7))
    targets = torch.randint(0, 31, tokens.shape)
    weights = torch.tensor([[1, -0.3, 0, 0.7, -1, 0.4, 0.2]])
    for _ in range(2):
        optimizer.zero_grad()
        other.zero_grad()
        actual = model.target_logprobs(tokens, targets, (3, 4), head_chunk_size=2)
        logits = oracle(tokens, (3, 4))
        expected = logits.log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1)
        torch.testing.assert_close(actual, expected)
        (actual * weights).sum().backward()
        (expected * weights).sum().backward()
        for a, b in zip(selected, reference, strict=True):
            torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=3e-5)
        optimizer.step()
        other.step()
        torch.testing.assert_close(
            model.state_dict(), oracle.state_dict(), atol=2e-6, rtol=3e-5
        )


def test_llama_logprob_path_captures_loss_and_outputs_from_the_same_graph():
    torch.manual_seed(206)
    model = Llama(Config(1, 16, 2, 1, 32, 31, max_seq_len=8))
    tokens = torch.randint(0, 31, (1, 5))
    targets = torch.randint(0, 31, tokens.shape)
    weights = torch.tensor([[1, -0.2, 0.3, 0, 0.4]])

    def objective(model, inputs, fields, controls):
        logprobs = model.target_logprobs(
            inputs["tokens"], fields["targets"], head_chunk_size=2
        )
        return -(logprobs * fields["weights"]).sum(), {"logprobs": logprobs}

    args = [{"tokens": tokens}, {"targets": targets, "weights": weights}, {}]
    captured = capture_training_objective(model, objective, args)
    loss, logprobs = captured.exported.exported_program.module()(*args)
    expected = F.cross_entropy(
        model(tokens).float().reshape(-1, 31), targets.reshape(-1), reduction="none"
    )
    torch.testing.assert_close(logprobs.reshape(-1), -expected)
    torch.testing.assert_close(loss, (expected * weights.reshape(-1)).sum())
    gradients = torch.autograd.grad(loss, tuple(model.parameters()))
    oracle = torch.autograd.grad(
        (expected * weights.reshape(-1)).sum(), tuple(model.parameters())
    )
    for a, b in zip(gradients, oracle, strict=True):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=3e-5)
