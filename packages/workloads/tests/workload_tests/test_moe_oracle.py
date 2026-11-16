"""Independent expert products preserve precision and cover both loss terms."""

from copy import deepcopy

import pytest
import torch

from shadowspill.training.definitions import load_registry
from workload_tests._moe_oracle import WeightGradients, experts
from workload_tests.test_moe_definitions import settings
from workload_tests.test_moe_objective import independent, physical


def model(lora, dtype):
    torch.manual_seed(69)
    requested = settings("qwen3moe", lora=lora, dtype=dtype)
    result = (
        load_registry(["shadowspill-workloads"])
        .resolve_model(requested.definition, settings=requested)
        .create()
    )
    if lora:
        with torch.no_grad():
            for name, parameter in result.named_parameters():
                if name.endswith(("lora_b", "lora_gate_up_b", "lora_down_b")):
                    parameter.normal_(std=0.02)
    return result


@pytest.mark.parametrize("lora", [False, True])
def test_fp32_all_parameter_products_match_autograd_across_contributions(lora):
    actual = model(lora, "float32")
    reference = deepcopy(actual)
    observer = WeightGradients(actual)
    for tokens, coefficient in (([1, 2, 3], 0.07), ([4, 5], 0.03)):
        inputs, fields = physical(tokens, [len(tokens)], weighted=True)
        loss = independent(
            actual,
            inputs,
            fields,
            coefficient,
            include_padding=False,
            router_observer=observer.router,
        )[0]
        loss.backward()
        independent(reference, inputs, fields, coefficient, include_padding=False)[
            0
        ].backward()
    expected = {n: p.grad for n, p in reference.named_parameters() if p.requires_grad}
    assert observer.values.keys() == expected.keys()
    for name, value in expected.items():
        torch.testing.assert_close(
            observer.values[name], value, atol=3e-6, rtol=3e-5, msg=name
        )
    observer.close()


@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_explicit_expert_forward_matches_eager_cpu_routing_and_empty_experts(
    lora, dtype
):
    module = model(lora, dtype).blocks[0].moe
    torch.manual_seed(83)
    hidden = torch.randn(2, 7, 32, dtype=getattr(torch, dtype))
    residual = torch.randn_like(hidden)
    # A wide selection margin leaves two experts empty without bypassing routing.
    hidden.abs_()
    with torch.no_grad():
        module.router.weight[0].fill_(0.1)
        module.router.weight[1].fill_(0.05)
        module.router.weight[2:].fill_(-0.1)
    expected = module(hidden, residual)[0]
    actual = experts(module, hidden, residual)
    # Per-expert traversal and grouped routing sum in different orders in FP32.
    # BF16 forward results for this wide-margin case agree exactly.
    tolerance = (
        {"rtol": 0, "atol": 0} if dtype == "bfloat16" else {"rtol": 1e-5, "atol": 1e-7}
    )
    torch.testing.assert_close(actual, expected, **tolerance)


def test_bf16_expert_product_is_retained_before_leaf_rounding():
    module = model(False, "bfloat16").blocks[0].moe
    observer = WeightGradients(module)
    # Exercise the reduction hook directly at a linear boundary with an exact
    # BF16 representable input whose summed product is not BF16 representable.
    x = torch.zeros(4, module.w2.shape[-1], dtype=torch.bfloat16)
    x[:, 0] = torch.tensor([1, 1 / 256, 1 / 256, 1 / 256])
    y = torch.nn.functional.linear(x, module.w2[0])
    observer._observe("w2", 0, x, y)
    y.sum().backward()
    assert observer.values["w2"].dtype == torch.float32
    assert observer.values["w2"][0, 0, 0].item() == 1 + 3 / 256
    assert module.w2.grad[0, 0, 0].item() != observer.values["w2"][0, 0, 0].item()
    assert not observer.values["w2"][1:].any()
    observer.close()
