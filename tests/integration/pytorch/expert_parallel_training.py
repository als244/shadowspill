"""torchrun: accumulated FP32-state EP training against eager EP + torch AdamW.

Run save and recompute separately with --resolution 0 and --resolution 1. This
checks planner/optimizer ownership; the eager reference shares MLOps' EP kernels.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import timedelta
from fractions import Fraction
from pathlib import Path

import torch
import torch.distributed as dist

from shadowspill.planner import SearchOptions, StepDataOrdering
from shadowspill.planner.search.algorithms.pressurefit import PressureFit
from shadowspill.planner.search.algorithms.pressurefit.options import PressureFitOptions
from shadowspill.pytorch import ProfilingOptions
from shadowspill.training import Trainer
from shadowspill.training.backends import ShadowSpill
from workloads.recipes.text.expert_parallel import experiment


def dense(value):
    value = value.detach()
    convert = getattr(value, "dequantize", None)
    if type(value) not in (torch.Tensor, torch.nn.Parameter) and callable(convert):
        return convert(dtype=torch.float32)
    return value.float()


def owned_reference(value, *, expert, rank, world_size):
    """EP experts are local; replicated parameters have flat optimizer shards."""
    value = value.detach().float().cpu()
    if expert:
        return value
    capacity = (value.numel() + world_size - 1) // world_size
    padded = torch.nn.functional.pad(
        value.reshape(-1), (0, capacity * world_size - value.numel())
    )
    return padded.narrow(0, rank * capacity, capacity)


def moment_error(actual, expected):
    """Moment comparisons expose reduction-scaling errors hidden by Adam."""
    assert actual.dtype == torch.float32
    assert actual.shape == expected.shape
    error = float(torch.linalg.vector_norm(actual - expected))
    scale = float(torch.linalg.vector_norm(expected))
    relative = error / max(scale, 1e-20)
    assert relative <= 0.03, (relative, error, scale)
    return relative


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, choices=(0, 1), required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    output = args.out / f"rank-{rank:05d}"
    output.mkdir(parents=True, exist_ok=True)
    dist.init_process_group("gloo", timeout=timedelta(seconds=900))
    try:
        policy = SearchOptions(
            workers=1,
            algorithm=PressureFit(
                PressureFitOptions(resolution_options=(Fraction(args.resolution),))
            ),
        )
        with ShadowSpill(
            device="auto",
            execution_gib=4,
            spill_gib=2,
            control_group=dist.group.WORLD,
            preparation_timeout=900,
            artifact_store=output / "artifacts",
            search_options=policy,
            orderings=lambda count: (StepDataOrdering(depth=2, breadth=2),),
            profiling_options=ProfilingOptions(
                conditioning_seconds=0,
                measurement_seconds=0,
                minimum_samples=3,
            ),
        ) as backend:
            recipe = experiment(
                device=backend.device,
                sequence_length=64,
                sequences_per_step=16,
                sequences_per_microbatch=4,
                preparation_timeout=900,
                config_overrides={
                    "n_layers": 2,
                    "d_model": 256,
                    "n_heads": 4,
                    "n_kv_heads": 2,
                    "head_dim": 64,
                    "n_experts": 16,
                    "top_k": 2,
                    "d_ff_expert": 256,
                    "vocab_size": 512,
                },
            )
            with recipe["context"]():
                source = recipe["model_factory"]()
                reference = recipe["model_factory"]().to(backend.device)
                ownership = recipe["distributed"](source)
                expert_names = {
                    name
                    for name, value in reference.named_parameters()
                    if any(value is item for item in reference.expert_parameters())
                }
                parameters = dict(reference.named_parameters())
                batches = recipe["candidates"]["4"]
                options = dict(
                    lr=0.001,
                    betas=(0.8, 0.95),
                    eps=1e-4,
                    weight_decay=0.01,
                )
                from mlops.optim import AdamW

                try:
                    with Trainer(
                        source,
                        objective=recipe["objective"],
                        microbatches=lambda values: values,
                        optimizer=AdamW,
                        optimizer_args={
                            **options,
                            "gradient_dtype": torch.float32,
                            "opt_state_dtype": torch.float32,
                        },
                        backend=backend,
                        distributed=ownership,
                        master_dtype=torch.float32,
                        grad_dtype=torch.float32,
                        shard_optimizer=True,
                    ) as trainer:
                        trainer.prepare(batches)
                        reference.load_state_dict(trainer.model.state_dict())
                        planned_state = (
                            trainer._execution.call._executor.optimizer_state
                        )
                        named_parameters = planned_state.optimizer_parameters
                        planned_names = {
                            id(value): name
                            for name, value in named_parameters.items()
                        }
                        masters = {
                            name: torch.nn.Parameter(dense(value).clone())
                            for name, value in parameters.items()
                        }
                        oracle = torch.optim.AdamW(
                            masters.values(), **options, foreach=False
                        )
                        observations = []
                        for step in range(args.steps):
                            reference.zero_grad(set_to_none=True)
                            oracle.zero_grad(set_to_none=True)
                            expected_loss = 0.0
                            for tokens, targets, lengths in batches:
                                loss, _ = recipe["objective"](
                                    reference,
                                    tokens.to(backend.device),
                                    targets.to(backend.device),
                                    lengths,
                                )
                                expected_loss += float(loss.detach())
                                loss.backward()
                            for name, parameter in parameters.items():
                                assert parameter.grad is not None, name
                                gradient = parameter.grad.float()
                                if name not in expert_names:
                                    dist.all_reduce(gradient, group=ownership.group)
                                masters[name].grad = gradient
                            oracle.step()
                            with torch.no_grad():
                                for name, parameter in parameters.items():
                                    parameter.copy_(masters[name])
                            observed = trainer.step(batches)
                            actual = trainer._execution.call.state_dict(
                                weights="compute"
                            )
                            checkpoint = trainer._execution.call.state_dict()
                            torch.testing.assert_close(
                                torch.tensor(observed.loss),
                                torch.tensor(expected_loss),
                                rtol=0.02,
                                atol=0.002,
                            )
                            errors = {}
                            for name, parameter in parameters.items():
                                wanted = dense(parameter).cpu()
                                value = dense(actual["model"][name]).cpu()
                                errors[name] = float((value - wanted).abs().max())
                                torch.testing.assert_close(
                                    value, wanted, rtol=0.03, atol=0.002
                                )
                            moment_errors, master_errors = [], []
                            saved_optimizer = checkpoint["optimizer"]
                            for live_group, saved_group in zip(
                                planned_state.optimizer.param_groups,
                                saved_optimizer["param_groups"],
                                strict=True,
                            ):
                                for parameter, index in zip(
                                    live_group["params"],
                                    saved_group["params"],
                                    strict=True,
                                ):
                                    name = planned_names[id(parameter)]
                                    state = saved_optimizer["state"][index]
                                    expected_state = oracle.state[masters[name]]

                                    def owned(value, *, name=name):
                                        return owned_reference(
                                            value,
                                            expert=name in expert_names,
                                            rank=rank,
                                            world_size=dist.get_world_size(),
                                        )

                                    for moment in ("exp_avg", "exp_avg_sq"):
                                        moment_errors.append(
                                            moment_error(
                                                state[moment],
                                                owned(expected_state[moment]),
                                            )
                                        )
                                    assert float(state["step"]) == step + 1
                                    master = checkpoint["masters"][name]
                                    wanted = owned(masters[name])
                                    assert master.dtype == torch.float32
                                    master_errors.append(
                                        float((master - wanted).abs().max())
                                    )
                                    torch.testing.assert_close(
                                        master, wanted, rtol=0.01, atol=0.0003
                                    )
                            assert moment_errors and master_errors
                            observations.append(
                                {
                                    "step": step + 1,
                                    "loss": observed.loss,
                                    "reference_loss": expected_loss,
                                    "maximum_parameter_error": max(errors.values()),
                                    "maximum_master_error": max(master_errors),
                                    "maximum_moment_relative_error": max(moment_errors),
                                }
                            )
                            print(
                                json.dumps({"rank": rank, **observations[-1]}),
                                flush=True,
                            )
                        trainer.diagnose(batches, directory=output / "trace", warmup=0)
                        trace = json.loads((output / "trace/step.json").read_text())
                        assert trace["summary"]["trace_complete"]
                        assert trace["runtime"]["callback_failures_after"] == 0
                        (output / "result.json").write_text(
                            json.dumps(
                                {
                                    "passed": True,
                                    "rank": rank,
                                    "world_size": dist.get_world_size(),
                                    "resolution": args.resolution,
                                    "steps": observations,
                                    "ordering": {"depth": 2, "breadth": 2},
                                    "expert_parameter_count": len(expert_names),
                                    "reference": (
                                        "eager EP with summed replicated gradients "
                                        "and torch FP32 AdamW; shares EP kernels"
                                    ),
                                },
                                indent=2,
                            )
                            + "\n"
                        )
                finally:
                    reference.close()
                    source.close()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
