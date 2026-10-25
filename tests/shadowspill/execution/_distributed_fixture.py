"""Small odd-sized model and independent, globally summed eager oracle."""

import copy

import torch
from torch import nn


class Network(nn.Module):
    def __init__(self, dtype=torch.float32, device="cpu"):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(7, 5, dtype=dtype, device=device),
                nn.Linear(5, 7, dtype=dtype, device=device),
            ]
        )

    def forward(self, x):
        return self.layers[1](self.layers[0](x).tanh())


def objective(model, x, *, labels):
    output = model(x)
    return (output.float() - labels.float()).square().mean(), {}


def examples(dtype):
    generator = torch.Generator().manual_seed(95)
    return [
        [
            (
                (torch.randn(4, 7, generator=generator).to(dtype),),
                {"labels": torch.randn(4, 7, generator=generator).to(dtype)},
            )
            for _ in range(2)
        ]
        for _ in range(2)
    ]


def oracle(kind, dtype):
    torch.manual_seed(71)
    model = Network(dtype=dtype)
    initial = copy.deepcopy(model.state_dict())
    calls = examples(dtype)
    masters = {
        name: nn.Parameter(p.float().detach().clone())
        for name, p in model.named_parameters()
    }
    optimizer = torch.optim.AdamW(masters.values(), lr=0.001, foreach=False)
    summed = {name: torch.zeros_like(master) for name, master in masters.items()}
    expected = []
    for invocation in range(3):
        accumulate = kind == "forward_backward" and invocation == 1
        if not accumulate:
            for gradient in summed.values():
                gradient.zero_()
        outputs = []
        for rank_calls in calls:
            local = []
            for args, kwargs in rank_calls[: 1 if kind == "forward" else 2]:
                if kind == "forward":
                    local.append(model(*args).detach().clone())
                    continue
                model.zero_grad(set_to_none=True)
                loss, _ = objective(model, *args, **kwargs)
                loss.backward()
                local.append(loss.detach().clone())
                for name, parameter in model.named_parameters():
                    summed[name].add_(parameter.grad.float())
            outputs.append(local)
        state = {}
        if kind == "step":
            for name, master in masters.items():
                master.grad = summed[name]
            optimizer.step()
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    parameter.copy_(masters[name])
            state.update(
                {
                    f"master.{name}": master.detach().clone()
                    for name, master in masters.items()
                }
            )
            state.update(
                {
                    f"optimizer.{name}.{field}": value.clone()
                    for name, master in masters.items()
                    for field, value in optimizer.state[master].items()
                }
            )
        elif kind == "forward_backward":
            state.update(
                {f"gradient.{name}": value.clone() for name, value in summed.items()}
            )
        state.update(
            {
                name: parameter.detach().clone()
                for name, parameter in model.named_parameters()
            }
        )
        expected.append(
            dict(
                outputs=outputs,
                state=state,
                accumulate=accumulate,
                evaluation=[model(*local[0][0]).detach().clone() for local in calls],
            )
        )
    return dict(initial=initial, examples=calls, expected=expected)
