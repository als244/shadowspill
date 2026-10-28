"""Ordinary eager EP forward/backward and independent FP32 AdamW updates."""

import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn

from benchmarking.quickstart.phases.common import experiment, read
from benchmarking.quickstart.phases.distributed import workers
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


def reference(root, request, group):
    setup = experiment(request, logical_rank=group.logical_rank)
    plan = Plan.load(root / "plan.ssp")
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
    expected = []
    with prepare_resources(
        plan.program, torch.device("cuda:0"), group.rank_map
    ) as resources:
        reduction_group = resources.groups.resolve(
            plan.program.parallelism.group("all_ranks")
        )
        for _ in range(request["steps"]):
            model.zero_grad(set_to_none=True)
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for tokens, labels, lengths in calls:
                loss, _ = setup["objective"](
                    model, tokens.cuda(), labels.cuda(), lengths
                )
                losses.append(loss.detach().cpu())
                loss.backward()
            # Stable order is essential: every participant must call the same SUM.
            for name, parameter in sorted(parameters.items()):
                gradient = parameter.grad.float()
                if id(parameter) not in experts:
                    dist.all_reduce(gradient, group=reduction_group)
                masters[name].grad = gradient
            optimizer.step()
            with torch.no_grad():
                for name, parameter in parameters.items():
                    parameter.copy_(masters[name])
            values = {name: value.cpu().clone() for name, value in payloads(model)}
            values.update(constants(plan.program))
            values.update(controls(plan.program))
            # A mastered parameter's BF16 compute copy is the model's own weight;
            # the Executor exports it under the compute prefix.
            values.update(
                {
                    "compute." + name: values[name]
                    for name in (n for n, _ in payloads(model))
                    if name.removesuffix("._data") in parameters
                }
            )
            values.update(
                {
                    "master." + name: value.detach().cpu().clone()
                    for name, value in masters.items()
                }
            )
            values.update(
                {
                    f"optimizer.{name}.{key}": value.cpu().clone()
                    for name, master in masters.items()
                    for key, value in optimizer.state[master].items()
                }
            )
            values["optimizer_group.0.lr"] = torch.tensor(rate)
            expected.append({"losses": losses, "values": values})
    model.close()
    torch.save(expected, root / f"reference-{group.logical_rank}.pt")


if __name__ == "__main__":
    root = Path(sys.argv[1])
    request = read(root / "request.json")
    with workers(request, {"world_size": 2}) as group:
        reference(root, request, group)
