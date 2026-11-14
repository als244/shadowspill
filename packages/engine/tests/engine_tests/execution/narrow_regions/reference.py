"""Ordinary compiled CUDA autograd, independent of captured tasks and Plans.

Normal Inductor fusion is part of the reference contract. CPU/eager FP16
derivatives have different rounding and can change signs near cancellation.
Each rank accumulates its own microbatches in FP16 before the two contributions
are summed. FP32 masters/Adam operate only after that low-precision sum.
"""

import hashlib
from contextlib import nullcontext
from pathlib import Path

import torch
from torch._inductor import config
from torch.utils._pytree import tree_map

from engine_tests.execution._distributed_regions import objective
from engine_tests.execution.narrow_regions.fixture import model_for, write


def run(directory, case, *, device="cuda", compiled=True):
    path = directory / "inputs.pt"
    protected = hashlib.sha256(path.read_bytes()).hexdigest()
    inputs = torch.load(path, weights_only=True)
    model = model_for(case).to(device)
    parameters = dict(model.named_parameters())
    assert set(parameters) == set(inputs["initial"])
    assert all(p.dtype == torch.float16 for p in parameters.values())
    assert {n for n, p in parameters.items() if p.requires_grad} == set(
        inputs["trainable"]
    )
    calls = tree_map(lambda value: value.to(device), inputs["examples"])

    def loss_function(*args, **kwargs):
        return objective(model, *args, **kwargs)[0]

    function = (
        torch.compile(loss_function, fullgraph=True, dynamic=False)
        if compiled
        else loss_function
    )
    repeats = []
    context = config.patch(emulate_precision_casts=False) if compiled else nullcontext()
    with context:
        for _ in range(2):
            with torch.no_grad():
                for name, parameter in parameters.items():
                    parameter.copy_(inputs["initial"][name])
            masters = {
                name: torch.nn.Parameter(parameter.detach().float().clone())
                for name, parameter in parameters.items()
                if parameter.requires_grad
            }
            optimizer = torch.optim.AdamW(masters.values(), lr=0.001, foreach=False)
            previous = None
            invocations = []
            for step, ranks in enumerate(calls):
                local_gradients, losses = [], []
                for local in ranks:
                    model.zero_grad(set_to_none=True)
                    local_losses = []
                    for args, kwargs in local:
                        loss = function(*args, **kwargs)
                        assert loss.dtype == torch.float32
                        loss.backward()
                        local_losses.append(loss.detach().clone())
                    local_gradients.append(
                        {
                            name: parameters[name].grad.detach().clone()
                            for name in masters
                        }
                    )
                    losses.append(local_losses)
                fresh = {
                    name: local_gradients[0][name] + local_gradients[1][name]
                    for name in masters
                }
                accumulating = case["kind"] == "forward_backward" and step == 1
                gradients = {
                    name: value + previous[name] if accumulating else value
                    for name, value in fresh.items()
                }
                assert all(
                    g.dtype == torch.float16 and torch.isfinite(g).all()
                    for g in gradients.values()
                )
                if case["kind"] == "step":
                    for name, master in masters.items():
                        master.grad = gradients[name].float()
                    optimizer.step()
                    with torch.no_grad():
                        for name, master in masters.items():
                            parameters[name].copy_(master)
                state = {name: p.detach().clone() for name, p in parameters.items()}
                if case["kind"] == "step":
                    state.update(
                        {
                            "master." + name: p.detach().clone()
                            for name, p in masters.items()
                        }
                    )
                    state.update(
                        {
                            f"optimizer.{name}.{field}": value.detach().clone()
                            for name, p in masters.items()
                            for field, value in optimizer.state[p].items()
                        }
                    )
                else:
                    state.update(
                        {"gradient." + name: g for name, g in gradients.items()}
                    )
                assert all(torch.isfinite(value).all() for value in state.values())
                invocations.append(
                    dict(
                        state=state,
                        gradients=gradients,
                        rank_gradients=local_gradients,
                        new_gradients=fresh,
                        losses=losses,
                        accumulate=accumulating,
                    )
                )
                previous = gradients
            repeats.append(
                tree_map(
                    lambda value: (
                        value.detach().cpu().clone()
                        if isinstance(value, torch.Tensor)
                        else value
                    ),
                    invocations,
                )
            )
    for first, second in zip(*repeats, strict=True):
        for kind in ("state", "gradients", "new_gradients"):
            assert all(torch.equal(v, second[kind][n]) for n, v in first[kind].items())
        assert all(
            torch.equal(a, b)
            for left, right in zip(first["losses"], second["losses"], strict=True)
            for a, b in zip(left, right, strict=True)
        )
    assert protected == hashlib.sha256(path.read_bytes()).hexdigest()
    result = dict(
        **inputs,
        expected=repeats[1],
        reference_contract="ordinary_normal_compiled_cuda"
        if compiled
        else "cpu_control",
    )
    torch.save(result, directory / "reference.pt")
    write(
        directory,
        "reference.json",
        dict(
            contract=result["reference_contract"],
            input_sha256=protected,
            repeats=2,
            repeat_bitwise=True,
            emulate_precision_casts=False if compiled else None,
            parameter_dtype="float16",
            gradient_dtype="float16",
            master_dtype="float32" if case["kind"] == "step" else None,
            independent_of_program_and_plan=True,
        ),
    )


if __name__ == "__main__":
    import json
    import sys

    directory = Path(sys.argv[1])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    cpu = sys.argv[2:] == ["--cpu-control"]
    assert not sys.argv[2:] or cpu
    run(
        directory,
        json.loads((directory / "case.json").read_text()),
        device="cpu" if cpu else "cuda",
        compiled=not cpu,
    )
