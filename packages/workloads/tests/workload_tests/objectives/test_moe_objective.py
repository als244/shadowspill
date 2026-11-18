"""Generic MoE objectives: model-independent composition and geometry-aware oracles."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from shadowspill.pytorch.capture.objective import capture_training_objective
from shadowspill.training.definitions import InputAdapter, ModelDefinition, Registry
from shadowspill.training.preparation.configuration.operations import OperationContract
from shadowspill_client import Objective
from shadowspill_client.protocol.definitions import DefinitionSource
from shadowspill_client.protocol.inputs import Capacity, InputSchema, TensorField
from shadowspill_client.protocol.operations import OperationKey
from shadowspill_client.text import text_batch
from shadowspill_workloads.definitions import moe_objective, token
from shadowspill_workloads.lora import configure_lora
from shadowspill_workloads.models.olmoe.config import Config as OLMoEConfig
from shadowspill_workloads.models.olmoe.implementations.mlops import Model as OLMoE
from workload_tests.models.test_qwen_moe import (
    Qwen3MoE,
    Qwen3MoEConfig,
    Qwen35MoE,
    Qwen35MoEConfig,
    tiny,
)


@pytest.fixture(params=["qwen3", "qwen35", "olmoe"])
def model(request):
    torch.manual_seed(41)
    if request.param == "olmoe":
        return OLMoE(OLMoEConfig(2, 32, 4, 4, 8, 4, 2, 16, 97, max_seq_len=32))
    cls, config = (
        (Qwen3MoE, Qwen3MoEConfig())
        if request.param == "qwen3"
        else (Qwen35MoE, Qwen35MoEConfig())
    )
    return cls(
        replace(
            tiny(config),
            n_layers=2,
            max_seq_len=32,
            full_attention_interval=2 if config.attention_gate else 1,
        )
    )


def resolve(model, *, include_padding=False, policy=False, weight_field="weights"):
    registry = Registry()
    token.register(registry)
    moe_objective.register(registry)
    registry.add_model(
        ModelDefinition(
            "external/moe@1",
            dict,
            lambda _: model,
            InputSchema(
                [
                    TensorField(
                        ("tokens",),
                        "int64",
                        ("position",),
                        batching="pack",
                        pack_axis=0,
                        minimum=0,
                        maximum=96,
                    )
                ],
                [
                    Capacity(
                        "token_capacity",
                        "tokens",
                        "sum",
                        ("tokens",),
                        0,
                        scheduled=True,
                    )
                ],
            ),
            capabilities=("target_logprobs", "router_statistics"),
            input_adapter=lambda config, layout: InputAdapter(padding_safe=True),
        ),
        source=DefinitionSource("external", "1", "a" * 64),
    )
    description = registry.resolve_model("external/moe@1")
    config = {
        "include_padding": include_padding,
        "participation_field": "response_mask" if policy else "loss_mask",
        "weight_field": weight_field,
    }
    if isinstance(model, OLMoE):
        config.update(layer_reduction="sum", normalize_top_k=True)
    settings = Objective.sum(
        {
            "primary": Objective(
                "shadowspill/clipped_policy@1"
                if policy
                else "shadowspill/cross_entropy@1"
            ),
            "balance": Objective("shadowspill/moe_balance@1", config=config),
        },
        weights={"balance": 0.001},
    )
    result = registry.resolve_objective(
        settings.definition,
        model=description,
        config=settings.config,
        defaults=settings.defaults,
    )
    return registry, description, result


def physical(tokens, lengths, *, weighted, size=8):
    actual = len(tokens)
    inputs = {
        "tokens": torch.tensor([tokens + [0] * (size - actual)]),
        "sequence_lengths": torch.tensor([*lengths, size - actual], dtype=torch.int32),
    }
    loss_inputs = {
        "target_tokens": (inputs["tokens"] + 1) % 97,
        "loss_mask": torch.arange(size)[None, :] < actual,
        "weights": torch.tensor(
            [
                [{1: 1, 2: -0.5, 3: 0, 4: 0.7, 5: 1.2}[t] for t in tokens]
                + [0] * (size - actual)
            ]
        )
        if weighted
        else torch.ones(1, size),
    }
    loss_inputs["router_mask"] = torch.arange(size)[None, :] < actual
    return inputs, loss_inputs


def independent(
    model,
    inputs,
    fields,
    coefficient,
    *,
    include_padding,
    policy=False,
    router_observer=None,
):
    """Full head and separately constructed router formula, with no loss helper."""
    logits = []

    def observe(module, args):
        routed = F.linear(args[0].float(), module.router.weight.float()).reshape(
            -1, model.config.n_experts
        )
        logits.append(routed)
        if router_observer is not None:
            router_observer(module, args[0], routed)

    hooks = [block.moe.register_forward_pre_hook(observe) for block in model.blocks]
    try:
        output = model(**inputs).float()
    finally:
        for hook in hooks:
            hook.remove()
    routes = torch.stack(logits)
    if not include_padding:
        routes = routes[:, fields["router_mask"].flatten()]
    if isinstance(model, OLMoE):
        frequency = (
            F.one_hot(
                routes.topk(model.config.top_k, dim=-1).indices, model.config.n_experts
            )
            .float()
            .mean((1, 2))
        )
        auxiliary = (
            model.config.n_experts * (frequency * routes.softmax(-1).mean(1)).sum()
        )
    else:
        routes = routes.flatten(0, 1)
        frequency = (
            F.one_hot(
                routes.topk(model.config.top_k, dim=-1).indices, model.config.n_experts
            )
            .float()
            .sum((0, 1))
            / routes.shape[0]
        )
        auxiliary = (
            model.config.n_experts * (frequency * routes.softmax(-1).mean(0)).sum()
        )
    valid = fields["response_mask" if policy else "loss_mask"]
    targets = torch.where(valid, fields["target_tokens"], 0)
    logprobs = output.log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1)
    if policy:
        ratio = (logprobs - fields["old_logprobs"]).exp()
        term = -torch.minimum(
            ratio * fields["advantages"], ratio.clamp(0.8, 1.2) * fields["advantages"]
        )
    else:
        term = -logprobs
    primary = torch.where(valid, term * fields["weights"], 0).sum()
    auxiliary_sum = auxiliary * torch.where(valid, fields["weights"], 0).sum()
    return primary + coefficient * auxiliary_sum, primary, auxiliary_sum


@pytest.mark.parametrize("split", [False, True])
def test_text_batch_normalizes_both_objective_losses_and_gradients(model, split):
    from workload_tests.definitions.test_token_definitions import bound

    batch = text_batch(
        [[1, 2, 3, 4], [4, 5, 6]],
        loss_masks=[[True, True, False, True], [True, True, True]],
    )
    count = 4  # Excludes the shifted first tokens and the masked target.
    _, description, objective = resolve(model)
    key = OperationKey("forward_backward" if split else "step", "metrics")
    for rows in [batch[:1], batch[1:]] if split else [batch]:
        with bound(description, objective, key, rows) as (binding, adapted):
            inputs, fields, controls = binding.arguments(adapted)
            lengths = [len(row.inputs["tokens"]) for row in rows]
            inputs = {
                "tokens": inputs["tokens"][None, :],
                "sequence_lengths": torch.tensor(
                    [*lengths, 8 - sum(lengths)], dtype=torch.int32
                ),
            }
            fields = {name: value[None, :] for name, value in fields.items()}
            actual = objective.computation(key)(model, inputs, fields, controls)
            unit_fields = {**fields, "weights": torch.ones_like(fields["weights"])}
            expected, primary, balance = independent(
                model, inputs, unit_fields, 0.001, include_padding=False
            )
            torch.testing.assert_close(actual.loss, expected / count)
            torch.testing.assert_close(
                actual.metrics["primary.loss_sum"], primary / count
            )
            torch.testing.assert_close(
                actual.metrics["balance.loss_sum"], balance / count
            )
            torch.testing.assert_close(
                torch.autograd.grad(actual.loss, tuple(model.parameters())),
                torch.autograd.grad(expected / count, tuple(model.parameters())),
                atol=3e-6,
                rtol=5e-4,
            )


@pytest.mark.parametrize("weight_field", [None, "balance_weights"])
def test_advanced_balancing_weights_can_be_independent(model, weight_field):
    _, _, objective = resolve(model, weight_field=weight_field)
    inputs, fields = physical([1, 2, 3, 4, 5], [3, 2], weighted=True)
    fields["loss_mask"][0, 0] = False
    controls = {name: torch.tensor(value) for name, value in objective.defaults.items()}
    if weight_field is not None:
        fields[weight_field] = torch.full_like(fields["weights"], 0.25)
        # Masked invalid values must not leak into the auxiliary objective.
        fields[weight_field][~fields["loss_mask"]] = float("nan")
    actual = objective.computation()(model, inputs, fields, controls)
    _, primary, _ = independent(model, inputs, fields, 0.001, include_padding=False)
    uniform = {**fields, "weights": torch.ones_like(fields["weights"])}
    _, _, balance = independent(model, inputs, uniform, 0.001, include_padding=False)
    balance = balance * (1 if weight_field is None else 0.25)
    torch.testing.assert_close(actual.metrics["balance.loss_sum"], balance)
    torch.testing.assert_close(actual.loss, primary + 0.001 * balance)
    torch.testing.assert_close(
        torch.autograd.grad(actual.loss, tuple(model.parameters())),
        torch.autograd.grad(primary + 0.001 * balance, tuple(model.parameters())),
        atol=3e-6,
        rtol=5e-4,
    )


@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("include_padding", [False, True])
def test_each_geometry_matches_independent_loss_all_gradients_and_two_updates(
    model, lora, split, include_padding
):
    if lora:
        model = configure_lora(
            model,
            rank=2,
            alpha=2,
            targets=[
                "blocks.*.attn.w*" if isinstance(model, OLMoE) else "blocks.*.mixer.w*",
                "blocks.*.moe",
            ],
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
    _, _, objective = resolve(model, include_padding=include_padding)
    compute = objective.computation(
        OperationKey("forward_backward" if split else "step", "logprobs")
    )
    calls = []
    hook = model.blocks[0].register_forward_hook(lambda *args: calls.append(1))
    for coefficient in (0.001, 0.07):
        controls = {
            name: torch.tensor(value)
            for name, value in objective.controls.resolve(
                defaults=objective.defaults, overrides={"balance.weight": coefficient}
            ).items()
        }
        actual_optimizer.zero_grad()
        expected_optimizer.zero_grad()
        for inputs, fields in batches:
            result = compute(model, inputs, fields, controls)
            loss, primary, auxiliary = independent(
                reference, inputs, fields, coefficient, include_padding=include_padding
            )
            torch.testing.assert_close(result.loss, loss, atol=4e-6, rtol=3e-5)
            torch.testing.assert_close(result.metrics["primary.loss_sum"], primary)
            torch.testing.assert_close(result.metrics["balance.loss_sum"], auxiliary)
            assert result.metrics["balance.valid_tokens"] == fields["loss_mask"].sum()
            assert not result.metrics["primary.logprobs"][
                ~fields["loss_mask"].flatten()
            ].any()
            (result.loss * 0.3).backward()
            (loss * 0.3).backward()
        for actual, expected in zip(
            actual_parameters, expected_parameters, strict=True
        ):
            torch.testing.assert_close(actual.grad, expected.grad, atol=3e-6, rtol=5e-4)
        actual_optimizer.step()
        expected_optimizer.step()
        for actual, expected in zip(
            model.parameters(), reference.parameters(), strict=True
        ):
            torch.testing.assert_close(actual, expected, atol=3e-6, rtol=5e-4)
    assert len(calls) == 2 * len(batches)
    hook.remove()


@pytest.mark.parametrize("include_padding", [False, True])
def test_forward_metrics_keeps_composed_loss_against_independent_oracle(
    model, include_padding
):
    inputs, fields = physical([1, 2, 3, 4, 5], [3, 2], weighted=True)
    _, _, objective = resolve(model, include_padding=include_padding)
    key = OperationKey("forward", "metrics", "evaluation")
    contract = OperationContract(key, objective)
    controls = {
        name: torch.tensor(value)
        for name, value in contract.loss_params({"balance.weight": 0.07}).items()
    }
    model.eval()
    with torch.no_grad():
        actual = objective.computation(key)(model, inputs, fields, controls)
        expected, _, _ = independent(
            model, inputs, fields, 0.07, include_padding=include_padding
        )
    torch.testing.assert_close(actual.loss, expected, atol=4e-6, rtol=3e-5)
    assert set(actual.metrics) == set(contract.outputs.metrics)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_explicit_legacy_option_matches_existing_model_loss_and_gradients(model):
    inputs, fields = physical([1, 2, 3, 4, 5], [3, 2], weighted=False)
    _, _, objective = resolve(model, include_padding=True)
    controls = {
        name: torch.tensor(value)
        for name, value in objective.controls.resolve(
            defaults=objective.defaults, overrides={"balance.weight": 0.07}
        ).items()
    }
    result = objective.computation()(model, inputs, fields, controls)
    legacy = model.loss(
        inputs["tokens"],
        torch.where(fields["loss_mask"], fields["target_tokens"], -100),
        seq_lens=inputs["sequence_lengths"],
        reduction="sum",
        aux_coef=0.07,
    )
    torch.testing.assert_close(result.loss, legacy)
    parameters = tuple(model.parameters())
    torch.testing.assert_close(
        torch.autograd.grad(result.loss, parameters),
        torch.autograd.grad(legacy, parameters),
        atol=3e-6,
        rtol=5e-4,
    )


@pytest.mark.parametrize("include_padding", [False, True])
def test_all_supervision_masked_has_zero_loss_and_gradients(model, include_padding):
    inputs, fields = physical([1, 2, 3], [3], weighted=True)
    fields["loss_mask"].zero_()
    _, _, objective = resolve(model, include_padding=include_padding)
    result = objective.computation()(
        model,
        inputs,
        fields,
        {name: torch.tensor(value) for name, value in objective.defaults.items()},
    )
    assert result.loss.item() == 0
    assert all(
        not g.any() for g in torch.autograd.grad(result.loss, tuple(model.parameters()))
    )


def test_default_excludes_padding_but_includes_unsupervised_real_prompt_tokens(model):
    _, _, objective = resolve(model)
    assert (
        objective.definition.padding_safe
        and not objective.definition.physical_batch_loss
    )
    controls = {name: torch.tensor(value) for name, value in objective.defaults.items()}
    results, gradients = [], []
    for size in (8, 16):
        inputs, fields = physical([1, 2, 3, 4, 5], [3, 2], weighted=True, size=size)
        fields["loss_mask"][0, 0] = False  # Real prompt token, still routed.
        inputs["tokens"][~fields["router_mask"]] = size + 17
        result = objective.computation()(model, inputs, fields, controls)
        expected, _, balance = independent(
            model, inputs, fields, 0.001, include_padding=False
        )
        torch.testing.assert_close(result.loss, expected)
        torch.testing.assert_close(result.metrics["balance.loss_sum"], balance)
        results.append(result.loss)
        gradients.append(torch.autograd.grad(result.loss, tuple(model.parameters())))
    torch.testing.assert_close(*results, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(*gradients, atol=3e-6, rtol=5e-4)


def test_policy_and_balancing_share_evaluation_and_match_independent_gradients(model):
    inputs, fields = physical([1, 2, 3, 4, 5], [3, 2], weighted=True)
    fields["response_mask"] = fields.pop("loss_mask")
    fields["response_mask"][0, 0] = False
    fields["old_logprobs"] = torch.full((1, 8), -4.5)
    fields["advantages"] = torch.tensor([[1.0, -0.3, 0.0, 2.0, -1.2, 0, 0, 0]])
    _, _, objective = resolve(model, policy=True)
    controls = {name: torch.tensor(value) for name, value in objective.defaults.items()}
    actual = objective.computation()(model, inputs, fields, controls)
    expected, _, _ = independent(
        model, inputs, fields, 0.001, include_padding=False, policy=True
    )
    torch.testing.assert_close(actual.loss, expected, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(
        torch.autograd.grad(actual.loss, tuple(model.parameters())),
        torch.autograd.grad(expected, tuple(model.parameters())),
        atol=3e-6,
        rtol=5e-4,
    )


def test_forward_logprobs_needs_no_secondary_fields_and_skips_router_summary(
    model, monkeypatch
):
    _, _, objective = resolve(model)
    monkeypatch.setattr(
        model, "token_outputs", lambda *a, **kw: pytest.fail("router summary requested")
    )
    inputs, fields = physical([1, 2], [2], weighted=False)
    contract = objective.outputs["logprobs"].select(forward=True)
    assert contract.loss_inputs == (("target_tokens",),) and not contract.controls
    result = objective.computation(OperationKey("forward", "logprobs", "evaluation"))(
        model, inputs, {"target_tokens": fields["target_tokens"]}, {}
    )
    torch.testing.assert_close(
        result["primary.logprobs"],
        model.target_logprobs(**inputs, targets=fields["target_tokens"]).flatten(),
    )


def test_generic_registration_replaces_family_specific_objective(model):
    registry, description, _ = resolve(model)
    with pytest.raises(ValueError, match="not installed"):
        registry.resolve_objective(
            "shadowspill/qwen_moe_cross_entropy@1", model=description
        )
    for invalid in (
        {"include_padding": "yes"},
        {"normalize_top_k": 1},
        {"layer_reduction": "mystery"},
        {"participation_field": "other"},
        {"weight_field": ""},
        {"weight_field": 1},
    ):
        with pytest.raises((TypeError, ValueError)):
            registry.resolve_objective(
                "shadowspill/moe_balance@1", model=description, config=invalid
            )
    without = replace(
        description,
        definition=replace(description.definition, capabilities=("target_logprobs",)),
    )
    with pytest.raises(ValueError, match="capabilit"):
        registry.resolve_objective("shadowspill/moe_balance@1", model=without)


def test_materializer_masks_only_artificial_rows_for_default_router_statistics(model):
    from workload_tests.definitions.test_token_definitions import bound, records

    _, description, objective = resolve(model)
    with bound(
        description, objective, OperationKey("forward_backward", "metrics"), records()
    ) as (binding, adapted):
        _, fields, _ = binding.arguments(adapted)
        assert fields["router_mask"].tolist() == [True] * 5 + [False] * 3
        assert fields["loss_mask"].sum().item() == 4
        assert fields["router_mask"][1] and not fields["loss_mask"][1]


@pytest.mark.parametrize("include_padding", [False, True])
def test_composed_model_capture_reuses_graph_with_new_controls_and_masks(
    model, include_padding
):
    if isinstance(model, Qwen35MoE):
        pytest.skip(
            "hybrid capture requires CUDA FLA; CPU reference reads sequence lengths"
        )
    _, _, objective = resolve(model, include_padding=include_padding)
    inputs, fields = physical([1, 2, 3, 4, 5], [3, 2], weighted=True)
    controls = {name: torch.tensor(value) for name, value in objective.defaults.items()}
    captured = capture_training_objective(
        model, objective.computation(), [inputs, fields, controls]
    )
    graph = captured.exported.exported_program.module()
    for coefficient in (0.001, 0.07):
        controls["balance.weight"] = torch.tensor(coefficient)
        actual, *_ = graph(inputs, fields, controls)
        expected, _, _ = independent(
            model, inputs, fields, coefficient, include_padding=include_padding
        )
        torch.testing.assert_close(actual, expected, atol=4e-6, rtol=3e-5)
        torch.testing.assert_close(
            torch.autograd.grad(actual, tuple(model.parameters())),
            torch.autograd.grad(expected, tuple(model.parameters())),
            atol=3e-6,
            rtol=5e-4,
        )
        fields["loss_mask"][0, 1] = False
