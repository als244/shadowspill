"""Eight-GPU Qwen3-30B-A3B full training: 4K x 512 sequences per global update.

Launch with torchrun --standalone --nproc-per-node=8 -m
workloads.recipes.text.train_qwen3_distributed --data /path/to/qwen3-tokens.
"""

import argparse

import torch
from mlops.optim import AdamW

from shadowspill.training import DataParallel, reset_parameters
from shadowspill.training._legacy_trainer import Trainer
from shadowspill.training.backends import ShadowSpill
from shadowspill.training.schedules import WarmupCosine
from workloads.models.catalog import build_on_meta
from workloads.recipes.text.data.fixed import TokenUpdates
from workloads.recipes.text.objectives import next_token_prediction


def main(data_directory):
    steps = 5
    # Trainer creates Gloo for coordination, then NCCL for tensor collectives.
    # Weight fetches all-gather; gradient evictions reduce-scatter.
    parallel = DataParallel(shard_weights=True, shard_gradients=True, timeout=7200)
    if parallel.world_size != 8:
        raise ValueError("launch this script with eight torchrun processes")
    # Each rank reads 64 distinct sequences; losses use global token normalization.
    # Search sizes from 1 to 32 sequences per microbatch, and all factor orderings.
    data = TokenUpdates(
        data_directory,
        sequence_length=4096,
        sequences_per_step=64,
        rank=parallel.rank,
        world_size=parallel.world_size,
        max_sequences_per_microbatch=32,
    )
    model = build_on_meta("qwen3moe-30b-a3b", dtype="bfloat16")
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
        microbatch_candidates=data.candidates,
        backend=ShadowSpill(
            device="auto", execution_gib=77, spill_gib=96, preparation_timeout=7200
        ),
        distributed=parallel,
        shard_optimizer=True,  # Separately shard FP32 masters and AdamW moments.
    )
    torch.manual_seed(0)
    # Allocate sharded storage, initialize randomly, search sizes/orderings, admit.
    trainer.prepare(data.example(), initialize=reset_parameters)
    trainer.fit(data, steps=steps, run_dir="runs/qwen3-dp8")
    trainer.close()  # Release callables, pools, then the owned process groups.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", required=True, help="Qwen3-tokenized corpus directory"
    )
    main(parser.parse_args().data)
