"""Ordinary eager updates with explicit accumulation and optional master weights."""

import torch
from torch import nn


class Network(nn.Module):
    def __init__(self, *, mixed=False):
        super().__init__()
        self.first = nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)
        self.second = nn.Linear(
            8, 4, bias=False, dtype=torch.float32 if mixed else torch.bfloat16
        )

    def forward(self, features):
        hidden = self.first(features).relu().to(self.second.weight.dtype)
        return self.second(hidden)


def objective(model, features):
    return model(features).float().square().mean()


def wide_gradients(model, gradient_dtype):
    """Observe linear cotangents before autograd rounds weight.grad to BF16.

    ShadowSpill writes these GEMM results directly at the requested gradient
    dtype. The oracle evaluates that multiplication from ordinary eager inputs
    and cotangents, independently of capture or graph transformations.
    """
    values = {}

    def forward(layer, args, output):
        features = args[0].detach().float()

        def backward(cotangent):
            values[id(layer.weight)] = cotangent.t().float() @ features

        output.register_hook(backward)

    for layer in model.modules():
        if (
            isinstance(layer, nn.Linear)
            and layer.weight.dtype == torch.bfloat16
            and gradient_dtype == torch.float32
        ):
            layer.register_forward_hook(forward)
    return values


def snapshot(model, stepped, optimizer):
    result = {}
    for (name, weight), parameter in zip(
        model.named_parameters(), stepped, strict=True
    ):
        result[name] = weight.detach().cpu().clone()
        if parameter is not weight:
            result[f"master.{name}"] = parameter.detach().cpu().clone()
        result.update(
            {
                f"optimizer.{name}.{key}": value.detach().cpu().clone()
                for key, value in optimizer.state[parameter].items()
            }
        )
    for key in ("lr", "betas", "eps", "weight_decay", "rounding_salts"):
        value = optimizer.param_groups[0][key]
        entries = enumerate(value) if isinstance(value, tuple) else [(None, value)]
        for index, tensor in entries:
            if isinstance(tensor, torch.Tensor):
                suffix = "" if index is None else f".{index}"
                result[f"optimizer_group.0.{key}{suffix}"] = tensor.cpu().clone()
    return result


def oracle(model, examples, optimizer_factory, *, master_dtype, gradient_dtype):
    with torch.no_grad():
        for weight in model.parameters():
            weight.fill_(0.125)
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    model.cuda()
    stepped = [
        nn.Parameter(weight.detach().to(master_dtype))
        if master_dtype is not None and weight.dtype != master_dtype
        else weight
        for weight in model.parameters()
    ]
    optimizer = optimizer_factory(stepped)
    gradients = wide_gradients(model, gradient_dtype)
    expected = []
    for _ in range(3):
        sums = [torch.zeros_like(p, dtype=gradient_dtype) for p in stepped]
        losses = []
        for args, kwargs in examples:
            model.zero_grad(set_to_none=True)
            loss = objective(model, *(a.cuda() for a in args), **kwargs)
            loss.backward()
            losses.append((loss.detach().cpu(), None))
            for total, weight in zip(sums, model.parameters(), strict=True):
                total.add_(gradients.get(id(weight), weight.grad).to(gradient_dtype))
        for parameter, gradient in zip(stepped, sums, strict=True):
            parameter.grad = gradient.to(parameter.dtype)
        optimizer.step()
        with torch.no_grad():
            for weight, parameter in zip(model.parameters(), stepped, strict=True):
                if parameter is not weight:
                    weight.copy_(parameter)
        expected.append((losses, snapshot(model, stepped, optimizer)))
    return dict(
        initial=initial,
        examples=examples,
        expected=expected,
        hyperparams=[{}] * 3,
        accumulate=[False] * 3,
        restore=False,
        lazy_kernels=["_adamw_torch_kernel"],
    )
