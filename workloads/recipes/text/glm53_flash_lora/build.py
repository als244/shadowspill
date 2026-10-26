"""Build saved training/evaluation candidates on one representative GPU."""

import json
import sys
from functools import partial

import torch
from mlops.optim import AdamW

from shadowspill import Parallelism, build_forward, build_step
from workloads.recipes.phases import settings
from workloads.recipes.text.phases.candidates import batch_sizes
from workloads.recipes.text.precision import Precision

from .inputs import Inputs, loss
from .model import checkpoint, describe


def run(directory):
    root, options = settings(directory)
    precision = Precision(
        master_dtype="float32", grad_dtype="float32", opt_state_dtype="float32"
    )
    precision.apply()
    objective = partial(loss, head_chunk_size=options["head_chunk_size"])
    parallelism = Parallelism.dp(ranks_per_replica=options["ranks"])
    with checkpoint(options) as (model, source, _initialize):
        metadata = describe(options, model, source)
        (root / "model.json").write_text(json.dumps(metadata, indent=2) + "\n")
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=torch.tensor(options["learning_rate"]),
            **precision.optimizer_arguments(),
        )
        for size in batch_sizes(options):
            configured = {**options, "sequences_per_example": size}
            data = Inputs(configured, world_size=options["ranks"])
            if len(data.data) < options["steps"]:
                raise ValueError(
                    "corpus does not contain enough complete global updates"
                )
            examples = data.example()
            args, kwargs = examples["args"][0], examples["kwargs"][0]
            print(f"building batch-{size}", flush=True)
            training = build_step(
                model,
                optimizer=optimizer,
                objective=objective,
                example_args=args,
                example_kwargs=kwargs,
                num_examples=len(examples["args"]),
                parallelism=parallelism,
                **precision.plan_arguments(),
            )
            training.save(root / f"programs/batch-{size}/training.ssp")
            if options["eval_every"]:
                validation = Inputs(
                    configured, split="val", world_size=options["ranks"]
                )
                if len(validation.data) < options["eval_updates"]:
                    raise ValueError(
                        "validation corpus is shorter than requested evaluation"
                    )
                evaluation = build_forward(
                    model.eval(),
                    forward_fn=objective,
                    example_args=args,
                    example_kwargs=kwargs,
                    parallelism=parallelism,
                )
                evaluation.save(root / f"programs/batch-{size}/evaluation.ssp")
                model.train()


if __name__ == "__main__":
    run(sys.argv[1])
