"""Small eager references for saved training with lazy operators and buffers."""

import copy

import torch
from torch import nn


class Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(8, 4))
        self.head = nn.Linear(4, 2)
        self.register_buffer("offset", torch.randn(4), persistent=False)
        self.custom_ops = True

    def forward(self, tokens):
        if self.custom_ops:
            import mlops

            hidden = mlops.embedding(tokens, self.weight)
        else:
            hidden = torch.nn.functional.embedding(tokens, self.weight)
        return self.head(hidden + self.offset)


def squared_error(model, features, targets, *, normalizer):
    return (model(features) - targets).square().sum() / normalizer


def token_loss(model, tokens, targets, *, normalizer):
    return (
        model.loss(
            tokens.reshape(1, -1),
            targets.reshape(1, -1),
            seq_lens=(tokens.shape[1],) * tokens.shape[0],
            reduction="sum",
        )
        / normalizer
    )


def fixture(kind):
    torch.manual_seed(720)
    if kind == "llama":
        from workloads.models.llama3.config import Config
        from workloads.models.llama3.implementations.mlops import Model
        from workloads.models.llama3.implementations.pytorch import Model as Reference

        config = Config(2, 128, 4, 2, 256, 128, max_seq_len=32)
        model, reference = Model(config), Reference(config)
        reference.load_state_dict(model.state_dict())
        inputs = (torch.randint(0, 128, (4, 16)), torch.randint(0, 128, (4, 16)))
        objective = token_loss
    else:
        model = (
            Embedding()
            if kind == "embedding"
            else nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 2))
        )
        reference = copy.deepcopy(model)
        if kind == "embedding":
            reference.custom_ops = False
        inputs = (
            torch.randint(0, 8, (8,)) if kind == "embedding" else torch.randn(8, 4),
            torch.randn(8, 2),
        )
        objective = squared_error
    initial = {
        name: value.detach().clone()
        for name, value in (*model.named_parameters(), *model.named_buffers())
    }
    options = {"normalizer": inputs[1].numel()}
    calls = [
        (tuple(value[start : start + 2].clone() for value in inputs), options)
        for start in range(0, len(inputs[0]), 2)
    ]
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    expected = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = objective(reference, *inputs, **options)
        loss.backward()
        optimizer.step()
        expected.append((float(loss.detach()), copy.deepcopy(reference.state_dict())))
    return model, objective, dict(initial=initial, calls=calls, expected=expected)
