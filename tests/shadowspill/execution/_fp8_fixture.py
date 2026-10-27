"""Frozen FP8 payload/scales with independent dense-reference bias updates."""

import copy

import torch
from mlops.block_fp8 import BlockFP8Weight
from torch import nn
from torch.nn import functional as F

from tests.shadowspill.execution._oracle import optimizer_values


class Network(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        data = (torch.arange(128 * 128, device=device) % 8).reshape(1, 128, 128)
        self.weight = nn.Parameter(
            BlockFP8Weight(
                data.to(torch.float8_e4m3fn),
                torch.full((1, 1, 1), 0.125, device=device),
            ),
            requires_grad=False,
        )
        self.bias = nn.Parameter(torch.zeros(128, device=device))

    def forward(self, x):
        return F.linear(x, self.weight.dequantize(dtype=torch.float32)[0], self.bias)

    predict = forward


def objective(model, x):
    return model(x).square().mean() / 4, {}


def oracle(*, min_shard_bytes=0):
    source = Network()
    initial = {
        "weight._data": source.weight._data.clone(),
        "weight._scale": source.weight._scale.clone(),
        "bias": source.bias.detach().clone(),
    }
    # CPU reference explicitly expands the saved FP8 values/scales. It neither
    # calls the provider's dequantization kernel nor shares compiled task code.
    model = nn.Linear(128, 128)
    model.weight = nn.Parameter(
        source.weight._data.float()[0] * source.weight._scale.item(),
        requires_grad=False,
    )
    model.bias.data.zero_()
    generator = torch.Generator().manual_seed(100)
    calls = [
        [((torch.randn(3, 128, generator=generator),), {}) for _ in range(2)]
        for _ in range(2)
    ]
    optimizer = torch.optim.AdamW([model.bias], lr=0.001, foreach=False)
    expected = []
    for invocation in range(3):
        optimizer.zero_grad(set_to_none=True)
        outputs = []
        for examples in calls:
            losses = []
            for args, _ in examples:
                loss, _ = objective(model, *args)
                loss.backward()
                losses.append(loss.detach().clone())
            outputs.append(losses)
        metrics = (
            {
                "bias": {
                    "param_norm": model.bias.detach().norm(),
                    "grad_norm": model.bias.grad.norm(),
                }
            }
            if invocation != 1
            else {}
        )
        optimizer.step()
        expected.append(
            dict(
                outputs=outputs,
                state={
                    **copy.deepcopy(initial),
                    "bias": model.bias.detach().clone(),
                    **optimizer_values(model, optimizer),
                },
                accumulate=False,
                parameter_metrics=metrics,
                evaluation=[
                    model(*examples[0][0]).detach().clone() for examples in calls
                ],
            )
        )
    return dict(
        initial=initial,
        examples=calls,
        expected=expected,
        precision="float32",
        exact_names=("weight._data", "weight._scale"),
        # One scale element split across two ranks: logical rank one owns padding.
        empty_by_rank={1: ("weight._scale",)} if min_shard_bytes == 0 else {},
    )
