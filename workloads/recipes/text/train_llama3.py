"""Llama 3 8B full training: 4K tokens x 16 sequences per optimizer update."""

import argparse

import torch
from mlops.optim import AdamW

from shadowspill.planner import StepDataOrdering
from shadowspill.training import reset_parameters
from shadowspill.training._legacy_trainer import Trainer
from shadowspill.training.backends import ShadowSpill
from shadowspill.training.schedules import WarmupCosine
from workloads.models.catalog import build_on_meta
from workloads.recipes.text.fixed import TokenUpdates
from workloads.recipes.text.objectives import next_token_prediction


def main(data_directory):
    steps = 5
    # One optimizer update contains 16 sequences, split into four microbatches.
    data = TokenUpdates(data_directory, sequence_length=4096, sequences_per_step=16)
    model = build_on_meta("llama3-8b", dtype="bfloat16")
    trainer = Trainer(
        model,
        objective=next_token_prediction,
        optimizer=AdamW,
        optimizer_args={"opt_state_dtype": torch.float32},
        schedules={
            "lr": WarmupCosine(lr=3e-4, min_lr=3e-5, warmup_steps=1, total_steps=steps)
        },
        master_dtype=torch.float32,
        grad_dtype=torch.float32,
        microbatches=data.splitter(sequences_per_microbatch=4),
        backend=ShadowSpill(
            execution_gib=28,
            spill_gib=144,
            # One pass visits all four microbatches at each layer (breadth-first).
            orderings=[StepDataOrdering(depth=1, breadth=4)],
        ),
    )
    # Open pools and plan storage/recomputation with the fixed size and ordering.
    trainer.prepare(data.example(), initialize=reset_parameters)
    # Perform forward/backward/AdamW updates, advancing the LR schedule each step.
    trainer.fit(data, steps=steps, run_dir="runs/llama3-8b")
    trainer.close()  # Release the callable and the pools this trainer opened.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", required=True, help="Llama-tokenized corpus directory"
    )
    main(parser.parse_args().data)
