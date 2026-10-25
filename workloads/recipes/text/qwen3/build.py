"""Build batch candidates on one GPU without creating distributed process groups."""

import sys

import torch
from mlops.dispatch import set_weight_gradient_dtype
from mlops.optim import AdamW

from shadowspill import Parallelism, build_step
from workloads.models.qwen3moe.implementations.mlops import Model
from workloads.recipes.phases import settings
from workloads.recipes.text.data.calls import Inputs
from workloads.recipes.text.objectives import normalized_next_token_prediction as loss

from .common import batch_sizes, model_config


def run(directory):
    root, options = settings(directory)
    config = model_config(options)
    with torch.device("meta"):
        model = Model(config).to(torch.bfloat16)
    set_weight_gradient_dtype(torch.float32)
    optimizer = AdamW(
        model.parameters(),
        lr=torch.tensor(options["learning_rate"]),
        opt_state_dtype=torch.float32,
        gradient_dtype=torch.float32,
    )
    parallelism = Parallelism.dp(ranks_per_replica=options["ranks"])
    for size in batch_sizes(options):
        source = Inputs(
            {**options, "sequences_per_example": size}, world_size=options["ranks"]
        )
        if source.data.vocab_size > config.vocab_size:
            raise ValueError("corpus vocabulary exceeds model vocabulary")
        examples = source.example()
        print(f"building batch-{size}", flush=True)
        program = build_step(
            model,
            optimizer=optimizer,
            objective=loss,
            example_args=examples["args"][0],
            example_kwargs=examples["kwargs"][0],
            num_examples=len(examples["args"]),
            parallelism=parallelism,
            master_dtype=torch.float32,
            grad_dtype=torch.float32,
        )
        program.save(root / f"programs/batch-{size}.ssp")


if __name__ == "__main__":
    run(sys.argv[1])
