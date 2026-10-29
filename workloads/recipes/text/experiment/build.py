"""Resolve a configured model/objective and save each geometry once."""

import functools
import json
import sys

import torch

from shadowspill import Parallelism, build_forward, build_step
from shadowspill.training import config
from shadowspill.training._inputs import ScaledObjective
from shadowspill.training._model import build_optimizer
from workloads.recipes.phases import settings

from . import targets
from .hyperparameters import scheduled_arguments
from .inputs import call_inputs, candidates, rank_options, ranks


def run(directory):
    root, options = settings(directory)
    target = targets.record(root, options)
    home = targets.artifacts(root, options)
    build_options = config.resolve(options.get("build", {}))
    # One worker per rank, the step's tokens shared among them; ``ranks``
    # workers shard weights, gradients and optimizer state by default.
    count = ranks(options)
    parallelism = build_options.get("parallelism")
    if parallelism is None and count > 1:
        parallelism = Parallelism.dp(ranks_per_replica=count)
        build_options["parallelism"] = parallelism
    elif parallelism is not None and parallelism.world_size != count:
        raise ValueError("build.parallelism must span exactly ranks workers")
    config.resolve(options.get("settings", []))
    torch.manual_seed(options.get("seed", 0))
    model = config.resolve(options["model"])
    data = config.resolve(options["data"])
    objective = ScaledObjective(
        functools.partial(
            config.resolve(options["objective"]),
            **config.resolve(options.get("objective_args", {})),
        )
    )
    rank = targets.first_rank(options, target["host"])
    example, _, geometries, skipped = candidates(rank_options(options, rank), data)
    manifest = {"candidates": {}, "skipped": skipped}
    for name, (tokens, count) in geometries.items():
        args = call_inputs(example, name)["args"][0]
        optimizer = build_optimizer(
            config.resolve(options["optimizer"]),
            model.parameters(),
            scheduled_arguments(
                config.resolve(options.get("optimizer_args", {})),
                options.get("schedules", {}),
            ),
        )
        program = build_step(
            model,
            optimizer=optimizer,
            objective=objective,
            example_args=args,
            num_examples=count,
            master_dtype=config.resolve(options.get("master_dtype")),
            grad_dtype=config.resolve(options.get("grad_dtype")),
            **build_options,
        )
        schedules = options.get("schedules", {})
        controls = {
            binding["name"]
            for binding in program.bindings["optimizer_bindings"]
            if binding["role"] == "hyperparameter"
        }
        if unknown := schedules.keys() - controls:
            raise ValueError(
                f"schedules must name saved tensor hyperparameters: {sorted(unknown)}; "
                f"available: {sorted(controls)}"
            )
        program.save(home / f"programs/{name}/training.ssp")
        if options.get("eval_every", 0):
            build_forward(
                model.eval(),
                forward_fn=objective,
                example_args=args,
                parallelism=parallelism,
            ).save(home / f"programs/{name}/evaluation.ssp")
            model.train()
        manifest["candidates"][name] = {
            "tokens_per_microbatch": tokens,
            "microbatches": count,
        }
    (home / "candidates.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    run(sys.argv[1])
