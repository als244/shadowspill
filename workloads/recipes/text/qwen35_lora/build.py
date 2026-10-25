"""Build each batch geometry once; ordering and selection belong to planning."""

import sys

import torch
from mlops.dispatch import set_weight_gradient_dtype
from mlops.optim import AdamW

from shadowspill import build_forward, build_step
from workloads.models.construction import build_on_meta
from workloads.models.qwen35moe.checkpoint import Qwen35MoECheckpoint
from workloads.models.qwen35moe.implementations.mlops import Model
from workloads.recipes.text.data.calls import Inputs
from workloads.recipes.text.objectives import normalized_next_token_prediction as loss
from workloads.recipes.text.phases import settings

from .candidates import batch_sizes


def run(directory):
    root, options = settings(directory)
    with Qwen35MoECheckpoint(
        options["checkpoint"], max_seq_len=options["sequence_length"]
    ) as checkpoint:
        config = checkpoint.config
    set_weight_gradient_dtype(torch.float32)
    for size in batch_sizes(options):
        source = Inputs({**options, "sequences_per_example": size})
        if source.data.vocab_size > config.vocab_size:
            raise ValueError("corpus vocabulary exceeds model vocabulary")
        examples = source.example()
        args, kwargs = examples["args"][0], examples["kwargs"][0]
        model = build_on_meta(
            Model,
            config=config,
            dtype="bfloat16",
            lora={"rank": options["lora_rank"], "alpha": options["lora_alpha"]},
        )
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=torch.tensor(options["learning_rate"]),
            opt_state_dtype=torch.float32,
            gradient_dtype=torch.float32,
        )
        print(f"building {size} sequences per example", flush=True)
        training = build_step(
            model,
            optimizer=optimizer,
            objective=loss,
            example_args=args,
            example_kwargs=kwargs,
            num_examples=len(examples["args"]),
            master_dtype=torch.float32,
            grad_dtype=torch.float32,
        )
        training.save(root / f"programs/batch-{size}/training.ssp")
        evaluation = build_forward(
            model.eval(),
            forward_fn=loss,
            example_args=args,
            example_kwargs=kwargs,
        )
        evaluation.save(root / f"programs/batch-{size}/evaluation.ssp")


if __name__ == "__main__":
    run(sys.argv[1])
