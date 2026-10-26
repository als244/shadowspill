"""Independent eager PyTorch results for repeated executor invocations."""

import copy

import torch
from torch.utils._pytree import tree_map

from tests.shadowspill.build.test_training import objective


def oracle(model, examples, kind):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        foreach=False,
        lr=torch.tensor(0.001) if kind == "step_tensor_lr" else 0.001,
    )
    initial = copy.deepcopy(model.state_dict())
    accumulate = [False, kind == "forward_backward", False]
    if kind == "backward_resume":
        accumulate = [True, False, True]
        for name, p in model.named_parameters():
            p.grad = torch.randn_like(p)
            initial[f"gradient.{name}"] = p.grad.clone()
    if kind == "step_resume":
        for args, kwargs in examples:
            objective(model, *args, **kwargs)[0].backward()
        optimizer.step()
        initial = copy.deepcopy(model.state_dict())
        initial.update(optimizer_values(model, optimizer))
    calls, expected, parameter_metrics = [], [], []
    for index in range(3):
        hyperparams = {}
        if kind == "step_tensor_lr":
            value = (0.002, 0.0005, 0.001)[index]
            optimizer.param_groups[0]["lr"].fill_(value)
            if index != 2:  # Omitted values use the captured default again.
                hyperparams["optimizer_group.0.lr"] = value
        calls.append(hyperparams)
        state = {}
        if kind == "forward":
            outputs = model(*examples[0][0], **examples[0][1])
        else:
            if not accumulate[index]:
                optimizer.zero_grad(set_to_none=True)
            outputs = [objective(model, *args, **kwargs) for args, kwargs in examples]
            for loss, _ in outputs:
                loss.backward()
            if kind.startswith("step"):
                parameter_metrics.append(
                    {
                        name: {
                            "param_norm": torch.linalg.vector_norm(
                                p.detach(), dtype=torch.float32
                            ),
                            "grad_norm": torch.linalg.vector_norm(
                                p.grad, dtype=torch.float32
                            ),
                        }
                        for name, p in model.named_parameters()
                        if p.grad is not None
                    }
                    if index != 1
                    else None
                )
                optimizer.step()
                state.update(optimizer_values(model, optimizer))
            else:
                state.update(
                    {
                        f"gradient.{n}": p.grad.clone()
                        for n, p in model.named_parameters()
                    }
                )
        if kind == "step_tensor_lr":
            state["optimizer_group.0.lr"] = optimizer.param_groups[0]["lr"].clone()
        state.update({n: p.detach().clone() for n, p in model.named_parameters()})
        expected.append(
            (
                tree_map(
                    lambda v: v.detach().clone() if isinstance(v, torch.Tensor) else v,
                    outputs,
                ),
                state,
            )
        )
    return dict(
        initial=initial,
        examples=examples,
        expected=expected,
        hyperparams=calls,
        accumulate=accumulate,
        restore=kind.endswith("resume"),
        parameter_metrics=parameter_metrics,
    )


def optimizer_values(model, optimizer):
    return {
        f"optimizer.{name}.{entry}": value.detach().clone()
        for name, p in model.named_parameters()
        for entry, value in optimizer.state[p].items()
    }
