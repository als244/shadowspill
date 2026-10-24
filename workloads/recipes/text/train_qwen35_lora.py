"""Qwen3.5-35B-A3B LoRA: 32K tokens x 8 sequences per optimizer update."""

import argparse

import torch
from mlops.optim import AdamW

from shadowspill.training._legacy_trainer import Trainer
from shadowspill.training.backends import ShadowSpill
from shadowspill.training.logging import Wandb
from shadowspill.training.schedules import WarmupCosine
from workloads.models.construction import build_on_meta
from workloads.models.qwen35moe.checkpoint import Qwen35MoECheckpoint
from workloads.models.qwen35moe.implementations.mlops import Model
from workloads.recipes.text.fixed import TokenUpdates
from workloads.recipes.text.objectives import next_token_prediction


def main(
    data_directory,
    checkpoint_directory,
    *,
    frozen_state_id,
    run_dir="runs/qwen35-lora",
    resume=None,
    wandb_project=None,
):
    steps = 5
    # Read next-token examples and supply candidate microbatch sizes.
    data = TokenUpdates(data_directory, sequence_length=32768, sequences_per_step=8)

    # Each validation item contains eight 32K sequences, just like a training item.
    # A fresh iterator evaluates the same held-out windows each time.
    def validation_data():
        return TokenUpdates(
            data_directory, split="val", sequence_length=32768, sequences_per_step=8
        )

    checkpoint = Qwen35MoECheckpoint(checkpoint_directory)
    model = build_on_meta(
        Model,
        config=checkpoint.config,
        dtype="bfloat16",
        lora={"rank": 32, "alpha": 32},
    )
    trainer = Trainer(
        model,
        objective=next_token_prediction,
        optimizer=AdamW,
        optimizer_args={"opt_state_dtype": torch.float32},
        schedules={
            "lr": WarmupCosine(lr=2e-4, min_lr=2e-5, warmup_steps=1, total_steps=steps)
        },
        master_dtype=torch.float32,
        grad_dtype=torch.float32,
        microbatch_candidates=data.candidates,
        backend=ShadowSpill(execution_gib=28, spill_gib=144),
        # Save trainable state only, referencing this immutable frozen base ID.
        frozen_state_id=frozen_state_id,
    )
    # Load the frozen base; on resume, restore saved adapters/optimizer/progress.
    trainer.prepare(data.example(), initialize=checkpoint.initialize, checkpoint=resume)
    # Local JSONL logging is always enabled; W&B is optional.
    logger = None
    if wandb_project is not None:
        logger = Wandb(
            project=wandb_project,
            run_dir=run_dir,
            config={
                "model": "Qwen3.5-35B-A3B",
                "lora_rank": 32,
                "sequence_length": 32768,
                "sequences_per_step": 8,
            },
        )
    trainer.fit(
        data,
        steps=steps,  # Target total updates, including any restored progress.
        run_dir=run_dir,
        log_every=1,  # Loss, LR and step timing go to metrics.jsonl and the logger.
        logger=logger,
        eval_data=validation_data,
        eval_every=2,  # Every two training updates, and also at the final update.
        eval_updates=2,  # Two validation updates: 16 sequences, 524,288 target tokens.
        checkpoint_every=2,  # Adapter masters once, optimizer, RNG and data position.
        keep_last=2,  # Retain the two newest checkpoints under run_dir/checkpoints.
    )
    trainer.close()  # Release the callable and its memory pools.
    if logger is not None:
        logger.close()  # Finish and flush this W&B run.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", required=True, help="Qwen3.5-tokenized corpus directory"
    )
    parser.add_argument("--checkpoint", required=True, help="Local HF base checkpoint")
    parser.add_argument(
        "--frozen-state-id",
        required=True,
        help="Immutable identity for the base checkpoint and model configuration",
    )
    parser.add_argument("--run-dir", default="runs/qwen35-lora")
    parser.add_argument("--resume", help="Trainer checkpoint directory to resume")
    parser.add_argument("--wandb-project", help="Also log to this W&B project")
    arguments = parser.parse_args()
    main(
        arguments.data,
        arguments.checkpoint,
        frozen_state_id=arguments.frozen_state_id,
        run_dir=arguments.run_dir,
        resume=arguments.resume,
        wandb_project=arguments.wandb_project,
    )
