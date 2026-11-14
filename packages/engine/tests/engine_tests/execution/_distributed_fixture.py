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
        # ReLU's exact derivative keeps the distributed qualification independent
        # of whether recomputation fuses an unrounded FP32 nonlinear derivative.
        return self.layers[1](self.layers[0](x).relu())


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
    linear_outputs = {}

    def remember_linear(module, args, output):
        output.retain_grad()
        linear_outputs[module] = (args[0].detach(), output)

    # The Program requests FP32 parameter gradients. A BF16 Parameter.grad
    # would round GEMM/reduction results before conversion; retain operands so
    # the eager oracle can form the unrounded FP32 weight and bias gradients.
    for module in model.modules():
        if isinstance(module, nn.Linear):
            module.register_forward_hook(remember_linear)
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
                    gradient = parameter.grad.float()
                    if name.endswith((".weight", ".bias")):
                        module = model.get_submodule(name.rsplit(".", 1)[0])
                        inputs, output = linear_outputs[module]
                        gradient = (
                            output.grad.float().T @ inputs.float()
                            if name.endswith(".weight")
                            else output.grad.float().sum(0)
                        )
                    summed[name].add_(gradient)
            outputs.append(local)
        state = {}
        parameter_metrics = {}
        if kind == "step":
            if invocation != 1:
                parameter_metrics = {
                    name: {
                        "param_norm": torch.linalg.vector_norm(
                            parameter.detach(), dtype=torch.float32
                        ),
                        "grad_norm": torch.linalg.vector_norm(
                            summed[name], dtype=torch.float32
                        ),
                    }
                    for name, parameter in model.named_parameters()
                }
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
                parameter_metrics=parameter_metrics,
                evaluation=[model(*local[0][0]).detach().clone() for local in calls],
            )
        )
    return dict(initial=initial, examples=calls, expected=expected)
