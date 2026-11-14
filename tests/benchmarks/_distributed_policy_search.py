"""A tied, sequence-coupled DP update with independently searched build policies."""

from dataclasses import replace
from functools import partial
from hashlib import sha256
from math import prod

import torch
from torch._inductor import config as inductor_config
from torch.utils._pytree import tree_map

from engine_tests.execution._distributed_regions import (
    Network,
    examples,
    objective,
    policies,
)
from engine_tests.execution._oracle import optimizer_values
from engine_tests.pytorch.chunking.test_regions import RegionChunks
from shadowspill.pytorch import (
    AxisChunks,
    ChunkRegions,
    GraphPairPolicy,
    RetentionPolicy,
    RetentionRule,
)
from shadowspill.pytorch.partition import IsolateOperations

GEOMETRIES = ((3, 7, 11), (5, 9, 13))


class InnerChunks:
    def __init__(self, projection, nonlinear):
        self.projection, self.nonlinear = projection, nonlinear

    def configure_stage(self, stage):
        axes = {
            index: domains["tokens"]
            for index, domains in enumerate(stage.input_domains)
            if "tokens" in domains
        }
        size = (
            self.nonlinear
            if any("silu" in op for op in stage.operators)
            else self.projection
        )
        return AxisChunks(size, axes, domain="tokens") if axes else None


class CandidateChunks(RegionChunks):
    def __init__(self, geometry):
        super().__init__(17, 2)
        self.projection, self.nonlinear, self.outer = geometry

    def configure_stage(self, stage):
        selected = super().configure_stage(stage)
        if selected is None:
            return None
        if not any("silu" in op for op in stage.operators):
            return replace(selected, chunk_size=self.projection)
        return replace(
            selected,
            chunk_size=self.outer,
            regions=ChunkRegions(
                IsolateOperations(
                    ("aten.silu.default", "aten.silu_backward.default"), base="whole"
                ),
                InnerChunks(self.projection, self.nonlinear),
            ),
        )


def policy_pairs():
    selected = RetentionPolicy(
        default="recompute", rules=(RetentionRule("aten._softmax.default", "retain"),)
    )
    return {
        "inputs_only": GraphPairPolicy(),
        "sequence_retained": GraphPairPolicy(recompute=selected),
        "sequence_only_save": GraphPairPolicy(save=selected),
    }


def initial_values(name, offset, count):
    seed = int.from_bytes(sha256(name.encode()).digest()[:4], "little")
    indices = torch.arange(offset, offset + count)
    return (indices * 19 + seed).remainder_(251).float().div_(125).sub_(1).mul_(0.1)


def fill_tensor(request, chunk):
    chunk.copy_(initial_values(request.name, request.offset, chunk.numel()))


def experiment(*, device, logical_rank=0):
    if logical_rank not in (0, 1):
        raise ValueError("logical_rank must be 0 or 1")
    calls = examples()[logical_rank]
    candidates, metadata = {}, {}
    for name, pair in policy_pairs().items():
        for geometry in GEOMETRIES:
            controls = policies(
                dict(
                    storage="sharded",
                    lora=False,
                    nested=True,
                    retain_sequence=False,
                    checkpoint_cuts=False,
                    lifetime_mode="replay_compact",
                )
            )
            label = name + "-chunks" + "_".join(map(str, geometry))
            candidates[label] = {
                "args": [args for args, _ in calls],
                "kwargs": [kwargs for _, kwargs in calls],
                "build_options": {
                    "chunking": replace(
                        controls["chunking"], chunking=CandidateChunks(geometry)
                    ),
                    "graph_pair_policy": pair,
                },
            }
            metadata[label] = {"chunks": geometry, "policy": pair.identity()}
    return {
        "model_factory": partial(
            Network, False, device="meta", lifetime_mode="replay_compact"
        ),
        "objective": objective,
        "optimizer": partial(torch.optim.AdamW, lr=0.001, foreach=False),
        "fill_tensor": fill_tensor,
        "build_options": {
            "parallelism": controls["parallelism"],
            "partition": controls["partition"],
            "grad_dtype": torch.float32,
            "master_dtype": None,
        },
        "context": partial(inductor_config.patch, emulate_precision_casts=True),
        "candidates": candidates,
        "metadata": {"candidates": metadata},
        "units_per_step": 2 * 2 * 2 * 17,
        "unit_label": "tokens",
    }


def reference_fixture():
    """Independent FP64 eager autograd, preserving exact actual FP32 inputs."""
    model = Network(False, dtype=torch.float64, lifetime_mode="replay_compact")
    initial = {
        name: initial_values(name, 0, prod(value.shape)).reshape(value.shape)
        for name, value in model.named_parameters()
    }
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(initial[name])
    calls = examples()
    reference_calls = tree_map(lambda value: value.double(), calls)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, foreach=False)
    expected = []
    for invocation in range(3):
        model.zero_grad(set_to_none=True)
        outputs = []
        for local in reference_calls:
            values = []
            for args, kwargs in local:
                loss, _ = objective(model, *args, **kwargs)
                loss.backward()
                values.append(loss.detach().clone())
            outputs.append(values)
        gradients = {
            name: value.grad.clone() for name, value in model.named_parameters()
        }
        model.zero_grad(set_to_none=True)
        flat = [call for local in reference_calls for call in local]
        global_loss, _ = objective(
            model,
            torch.cat([args[0] for args, _ in flat]),
            **{
                key: torch.cat([kwargs[key] for _, kwargs in flat])
                for key in flat[0][1]
            },
        )
        global_loss.backward()
        torch.testing.assert_close(
            global_loss, sum(sum(local) for local in outputs), atol=1e-12, rtol=1e-12
        )
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(
                parameter.grad, gradients[name], atol=1e-12, rtol=1e-10
            )
            parameter.grad.copy_(gradients[name])
        metrics = (
            {
                name: dict(
                    param_norm=value.detach().norm(), grad_norm=value.grad.norm()
                )
                for name, value in model.named_parameters()
            }
            if invocation != 1
            else {}
        )
        optimizer.step()
        state = {
            **optimizer_values(model, optimizer),
            **{
                name: value.detach().clone() for name, value in model.named_parameters()
            },
        }
        expected.append(
            dict(
                outputs=outputs,
                state=state,
                accumulate=False,
                parameter_metrics=metrics,
            )
        )
    return dict(
        initial=initial,
        examples=calls,
        expected=tree_map(
            lambda v: v.detach().float() if isinstance(v, torch.Tensor) else v, expected
        ),
        precision="float32",
        reference_dtype="float64",
        exact_names=[],
        trace=True,
        trace_invocations=[0, 1, 2],
    )
