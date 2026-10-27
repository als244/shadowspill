"""Independent replicas, shared dense parameters and distinct local model shards."""

import copy

import torch
from torch import nn

from shadowspill import Parallelism, ParameterStorage, Sharded
from tests.shadowspill.execution._oracle import optimizer_values


class Network(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.shared = nn.Linear(7, 5, bias=False, device=device)
        self.local = nn.Linear(5, 3, bias=False, device=device)
        self.register_buffer("counter", torch.tensor(0, device=device))

    def forward(self, x):
        self.counter.add_(1)
        return self.predict(x)

    def predict(self, x):
        return self.local(self.shared(x).relu())


def objective(model, x, *, labels):
    return (model(x) - labels).square().mean(), {}


def parallelism(mode):
    if mode == "cutoff":
        return Parallelism.dp(ranks_per_replica=2, min_shard_bytes=100)
    if mode == "diagnostics":
        return Parallelism.dp(ranks_per_replica=2)
    replicated = mode == "replicated"
    return Parallelism(
        ranks_per_replica=2,
        groups={"all_ranks": ((0, 1),), "local": ((0,), (1,))},
        default_parameter_storage=ParameterStorage(
            group="all_ranks",
            weights="replicated",
            gradients="replicated" if replicated else "sharded",
            optimizer="replicated" if replicated else "sharded",
        ),
        parameter_placements={}
        if replicated
        else {"local.weight": Sharded("all_ranks", 0)},
        parameter_storage={}
        if replicated
        else {"local.weight": ParameterStorage(group="local")},
    )


def oracle(mode):
    torch.manual_seed(318)
    models = [Network()]
    models.append(copy.deepcopy(models[0]))
    if mode == "local_experts":
        with torch.no_grad():
            models[1].local.weight.normal_(std=0.1)
    initial = [copy.deepcopy(model.state_dict()) for model in models]
    calls = [
        [((torch.randn(4, 7),), {"labels": torch.randn(4, 3)}) for _ in range(2)]
        for _ in models
    ]
    optimizers = [
        torch.optim.AdamW(model.parameters(), lr=0.001, foreach=False)
        for model in models
    ]
    expected = []
    for invocation in range(3):
        outputs = []
        for model, optimizer, examples in zip(models, optimizers, calls, strict=True):
            optimizer.zero_grad(set_to_none=True)
            local = []
            for args, kwargs in examples:
                loss, _ = objective(model, *args, **kwargs)
                loss.backward()
                local.append(loss.detach().clone())
            outputs.append(local)
        metrics = {}
        for name in ("shared.weight", "local.weight"):
            parameters = [model.get_parameter(name) for model in models]
            unique = name == "local.weight" and mode == "local_experts"
            if not unique:
                gradient = sum(parameter.grad for parameter in parameters)
                for parameter in parameters:
                    parameter.grad.copy_(gradient)
            measured = parameters if unique else parameters[:1]
            metrics[name] = {
                "param_norm": sum(p.detach().square().sum() for p in measured).sqrt(),
                "grad_norm": sum(p.grad.square().sum() for p in measured).sqrt(),
            }
        states = []
        for model, optimizer in zip(models, optimizers, strict=True):
            optimizer.step()
            state = {
                **copy.deepcopy(model.state_dict()),
                **optimizer_values(model, optimizer),
            }
            if mode == "diagnostics":
                state["optimizer_group.0.lr"] = torch.tensor(0.001)
                for key in state:
                    if key.endswith(".step"):
                        state[key] = state[key].to(torch.int64)
            states.append(state)
        expected.append(
            dict(
                outputs=outputs,
                states=states,
                accumulate=False,
                parameter_metrics=metrics if invocation != 1 else {},
                evaluation=[
                    model.predict(*examples[0][0]).detach().clone()
                    for model, examples in zip(models, calls, strict=True)
                ],
            )
        )
    return dict(
        initial_by_rank=initial,
        examples=calls,
        expected=expected,
        precision="float32",
        diagnostics=mode == "diagnostics",
    )
