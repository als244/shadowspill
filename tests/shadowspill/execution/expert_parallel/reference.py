"""Ordinary eager EP steps and independent FP32 AdamW updates, each taken from
the state ShadowSpill held before it and compared with the step it took.

The first step starts from this arm's own initialization, filled as the
Executor's is; every later step starts from the weights, masters and moments
the Executor exported after the one before. Both arms then route every token to
the same experts, and each comparison measures one step's arithmetic. From
weights that differ in their last bits, a token near a routing tie can take
another expert in one arm only and move whole experts' moments.
"""

import json
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn

from benchmarking.quickstart.phases.common import experiment, read
from benchmarking.quickstart.phases.distributed import workers
from qualification.numerical.tolerances import STEP_QUANTUM_DISAGREEMENTS
from shadowspill import Plan
from shadowspill.artifacts.tensors import TensorChunk
from shadowspill.execution.storage import dtype_of
from shadowspill.resources.preparation import prepare_resources


def payloads(model):
    # Every tensor the Program declares, as the build inventories them: the
    # parameters and all buffers, the non-persistent rotary tables included,
    # which state_dict() leaves out and the Executor fills and exports.
    tensors = {
        **dict(model.named_buffers(remove_duplicate=False)),
        **dict(model.named_parameters(remove_duplicate=False)),
    }
    for name, value in tensors.items():
        # Storage views, as state_dict() hands them out: filled and read in
        # place, never through autograd.
        if hasattr(value, "_data"):
            yield name + "._data", value._data.detach()
        else:
            yield name, value.detach()


def constants(program):
    """The Program's lifted constants, named as the Executor exports them.

    Mirrors shadowspill.execution.tensors: a step program scopes each
    example's constants by example index, a forward program does not.
    """

    bindings = program.bindings
    groups = bindings["root_inputs"]
    if bindings["kind"] == "forward":
        groups = [groups]
    values = {}
    for index, slots in enumerate(groups):
        sources = (
            bindings["example_root_sources"][index]
            if "example_root_sources" in bindings
            else bindings["root_sources"]
        )
        for slot in slots:
            source = sources[slot["leaf_index"]]
            if source["kind"] == "CONSTANT_TENSOR" and "value" in source:
                scope = "" if bindings["kind"] == "forward" else f"example_{index}."
                values["constant." + scope + source["name"]] = torch.tensor(
                    source["value"]
                )
    return values


def controls(program):
    """The optimizer's hyperparameter controls at the values the Program gives them.

    The Executor holds each one as a bounded CPU tensor and exports it beside
    the state; a run supplies only the learning rate, so every other control
    keeps its initializer.
    """

    values = {}
    for binding in program.bindings.get("optimizer_bindings", ()):
        initializer = binding.get("initialization") or {}
        if (
            binding["object_id"] is not None
            or binding["role"] != "hyperparameter"
            or initializer.get("kind") not in {"constant", "value"}
        ):
            continue
        values[binding["name"]] = torch.as_tensor(
            initializer["value"], dtype=dtype_of(binding)
        ).reshape(tuple(binding["shape"]))
    return values


def dense(value):
    convert = getattr(value, "dequantize", None)
    if type(value) not in (torch.Tensor, nn.Parameter) and callable(convert):
        return convert(dtype=torch.float32)
    return value.float()


def held(states, rank, after, name, numel):
    """The complete tensor ``name`` as the Executor held it after step ``after``:
    this rank's region when it spans the tensor, otherwise the ranks' regions,
    which must tile it, as a sharded master or moment's do."""
    own = states[rank][after]["tensors"].get(name)
    if own is not None and own[0] == 0 and own[1].numel() == numel:
        return own[1]
    regions = sorted(
        (
            state[after]["tensors"][name]
            for state in states
            if name in state[after]["tensors"]
        ),
        key=lambda region: region[0],
    )
    at = 0
    for offset, part in regions:
        assert offset == at, (name, after, offset, at)
        at += part.numel()
    assert at == numel, (name, after, at, numel)
    return torch.cat([part for _, part in regions])


def take(states, rank, after, model, masters, optimizer):
    """Give this arm the state the Executor held after step ``after``; a
    tensor the Executor does not hold cannot have moved, and keeps this arm's
    value. A mastered parameter's weights are one object with its compute
    copy, exported once under whichever name comes first."""
    names = set().union(*(state[after]["tensors"] for state in states))
    with torch.no_grad():
        for name, value in payloads(model):
            for exported in (name, "compute." + name):
                if exported in names:
                    found = held(states, rank, after, exported, value.numel())
                    value.copy_(found.reshape(value.shape))
                    break
        for name, master in masters.items():
            found = held(states, rank, after, "master." + name, master.numel())
            master.copy_(found.reshape(master.shape))
            moments = {
                key: held(
                    states, rank, after, f"optimizer.{name}.{key}", master.numel()
                )
                for key in ("exp_avg", "exp_avg_sq")
            }
            optimizer.state[master] = {
                "step": held(states, rank, after, f"optimizer.{name}.step", 1)
                .reshape(())
                .float()
                .clone(),
                **{
                    key: value.reshape(master.shape).to(master.device)
                    for key, value in moments.items()
                },
            }


def step(model, parameters, experts, masters, optimizer, objective, calls, group):
    """One accumulated step: BF16 .grad summed over the microbatches, the
    replicated gradients summed over the ranks in FP32, then FP32 AdamW."""
    model.zero_grad(set_to_none=True)
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for tokens, labels, lengths in calls:
        loss, _ = objective(model, tokens.cuda(), labels.cuda(), lengths)
        losses.append(loss.detach().cpu())
        loss.backward()
    # Stable order is essential: every participant must call the same SUM.
    for name, parameter in sorted(parameters.items()):
        gradient = parameter.grad.float()
        if id(parameter) not in experts:
            dist.all_reduce(gradient, group=group)
        masters[name].grad = gradient
    optimizer.step()
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(masters[name])
    return losses


def named_state(model, parameters, masters, optimizer, program, rate):
    """This arm's state, named as the Executor exports it."""
    result = {name: value.cpu().clone() for name, value in payloads(model)}
    result.update(constants(program))
    result.update(controls(program))
    # A mastered parameter's BF16 compute copy is the model's own weight;
    # the Executor exports it under the compute prefix.
    result.update(
        {
            "compute." + name: result[name]
            for name in (n for n, _ in payloads(model))
            if name.removesuffix("._data") in parameters
        }
    )
    result.update(
        {
            "master." + name: value.detach().cpu().clone()
            for name, value in masters.items()
        }
    )
    result.update(
        {
            f"optimizer.{name}.{key}": value.cpu().clone()
            for name, master in masters.items()
            for key, value in optimizer.state[master].items()
        }
    )
    result["optimizer_group.0.lr"] = torch.tensor(rate)
    return result


def compare(tensors, expected, rate):
    """This rank's state after a step against this arm's step from the same
    state, every failing tensor named at once; the worst moment error and the
    worst master error in learning rates."""
    # The two arms round the same gradient sums in different orders, so an
    # element whose gradient is near zero can carry opposite signs and move one
    # learning rate each way in the step: a master weight is held to that
    # bound, the rule the numerical gate applies to such weights per step.
    master_bound = STEP_QUANTUM_DISAGREEMENTS * rate
    # Name every tensor the reference lacks at once; the Executor exports the
    # whole Program, and the reference must account for all of it.
    missing = sorted(set(tensors) - set(expected))
    assert not missing, f"the reference has no values for {missing}"
    failures = []
    moment, master = 0.0, 0.0
    for name, (offset, actual) in tensors.items():
        wanted = expected[name].reshape(-1)[offset : offset + actual.numel()]
        if name.endswith((".exp_avg", ".exp_avg_sq")):
            assert actual.dtype == torch.float32
            relative = float((actual - wanted).norm() / wanted.norm().clamp_min(1e-20))
            moment = max(moment, relative)
            if relative > 0.03:
                failures.append(f"{name}: relative L2 {relative:.4f} above 0.03")
            continue
        mastered = name.startswith("master.")
        if mastered:
            master = max(master, float((actual - wanted).abs().max()) / rate)
        try:
            torch.testing.assert_close(
                actual,
                wanted.to(actual.dtype),
                rtol=0.0 if mastered else 0.03,
                atol=master_bound if mastered else 0.002,
            )
        except AssertionError as error:
            failures.append(f"{name}: {error}")
    assert not failures, "\n".join(failures)
    assert any(name.endswith(".exp_avg") for name in tensors)
    assert any(name.startswith("master.") for name in tensors)
    return moment, master


def reference(root, request, group):
    setup = experiment(request, logical_rank=group.logical_rank)
    plan = Plan.load(root / "plan.ssp")
    rank = group.logical_rank
    states = [
        torch.load(root / f"state-{index}.pt", weights_only=True)
        for index in range(group.count)
    ]
    assert len(states[rank]) == request["steps"] + 1
    model = setup["model_factory"]().to_empty(device="cuda:0")
    with torch.no_grad():
        for name, value in payloads(model):
            source = torch.empty(value.shape, dtype=value.dtype, device="cpu")
            setup["fill_tensor"](
                TensorChunk(name, tuple(value.shape), value.dtype, 0, value.numel()),
                source.reshape(-1),
            )
            value.copy_(source)
    parameters = dict(model.named_parameters())
    experts = {id(p) for p in model.expert_parameters()}
    masters = {
        name: nn.Parameter(dense(p.detach()).clone()) for name, p in parameters.items()
    }
    rate = setup["hyperparams"]["optimizer_group.0.lr"]
    optimizer = torch.optim.AdamW(masters.values(), lr=rate, foreach=False)
    calls = setup["candidates"]["4"]["args"]
    records = []
    with prepare_resources(
        plan.program, torch.device("cuda:0"), group.rank_map
    ) as resources:
        reduction_group = resources.groups.resolve(
            plan.program.parallelism.group("all_ranks")
        )
        for index in range(1, len(states[rank])):
            # The first step starts from this arm's own values, so it checks
            # the Executor's initialization rather than adopting it.
            if index > 1:
                take(states, rank, index - 1, model, masters, optimizer)
            losses = step(
                model,
                parameters,
                experts,
                masters,
                optimizer,
                setup["objective"],
                calls,
                reduction_group,
            )
            taken = states[rank][index]
            for loss, wanted in zip(taken["losses"], losses, strict=True):
                torch.testing.assert_close(loss, wanted, rtol=0.02, atol=0.002)
            moment, master = compare(
                taken["tensors"],
                named_state(model, parameters, masters, optimizer, plan.program, rate),
                float(rate),
            )
            records.append(
                {
                    "step": index,
                    "loss": sum(float(v) for v in taken["losses"]),
                    "reference_loss": sum(float(v) for v in losses),
                    "maximum_moment_relative_error": moment,
                    "maximum_master_error_learning_rates": master,
                }
            )
            print(json.dumps({"rank": rank, **records[-1]}), flush=True)
    model.close()
    (root / f"result-{rank}.json").write_text(
        json.dumps({"passed": True, "steps": records}, indent=2) + "\n"
    )


if __name__ == "__main__":
    root = Path(sys.argv[1])
    request = read(root / "request.json")
    with workers(request, {"world_size": 2}) as group:
        reference(root, request, group)
