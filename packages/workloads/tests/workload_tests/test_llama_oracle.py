"""The independent weight reductions agree with ordinary wide autograd."""

import pytest
import torch
from mlops.lora import LoRAConfig
from mlops.lora.linear import LoRALinear

from shadowspill_workloads.models.shared.decoder import RMSNorm
from workload_tests._llama_oracle import WeightGradients


@pytest.mark.parametrize("kind", ["linear", "embedding", "norm", "lora"])
def test_explicit_weight_vjps_match_fp32_autograd_across_contributions(kind):
    torch.manual_seed(71)
    model = {
        "linear": lambda: torch.nn.Linear(4, 3, bias=False),
        "embedding": lambda: torch.nn.Embedding(7, 4),
        "norm": lambda: RMSNorm(4),
        "lora": lambda: LoRALinear(
            torch.nn.Linear(4, 3, bias=False),
            LoRAConfig(rank=2, alpha=3, factor_dtype=torch.float32),
        ),
    }[kind]()
    if kind == "lora":
        model.weight.requires_grad_(False)
        torch.nn.init.normal_(model.lora_b)
    observer = WeightGradients(model)
    for size in (5, 3):
        inputs = (
            torch.tensor([1, 1, 4, 3, 1][:size])
            if kind == "embedding"
            else torch.randn(size, 4)
        )
        output = model(inputs)
        output.backward(torch.randn_like(output))
    expected = {n: p.grad for n, p in model.named_parameters() if p.requires_grad}
    assert observer.values.keys() == expected.keys()
    for name, value in expected.items():
        torch.testing.assert_close(observer.values[name], value)
    observer.close()


def test_bf16_weight_leaf_does_not_round_the_observed_fp32_reduction():
    model = torch.nn.Linear(1, 1, bias=False).bfloat16()
    observer = WeightGradients(model)
    inputs = torch.tensor([[1.0], [1 / 256], [1 / 256], [1 / 256]]).bfloat16()
    model(inputs).sum().backward()
    assert observer.values["weight"].dtype == torch.float32
    assert observer.values["weight"].item() == 1 + 3 / 256
    assert model.weight.grad.item() != observer.values["weight"].item()
    observer.close()
