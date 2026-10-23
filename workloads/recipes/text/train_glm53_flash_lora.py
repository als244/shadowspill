"""GLM-5.3-Flash checkpoint LoRA on real text, with optional sharded DP.

Example: bash workloads/recipes/text/glm53_flash_lora_32k_8gpu.sh
--data DIR --checkpoint DIR
--frozen-state-id IMMUTABLE_BASE_ID --run-dir /scratch/glm-run

Prepare the corpus with the checkpoint's own tokenizer. Each rank defaults to
one independent 32K sequence per update. Frozen storage follows the checkpoint;
dense GEMMs use BF16, routed expert GEMMs use the preserved block-FP8 weights,
and trainable adapters use BF16 compute with FP32 masters, gradients and moments.
"""

import argparse
import json
import math
import os
import time
from fractions import Fraction
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from mlops.optim import AdamW

from shadowspill.planner import (
    GenericPlanningOptions,
    SearchOptions,
)
from shadowspill.planner.search.algorithms.pressurefit import (
    PressureFit,
    PressureFitOptions,
)
from shadowspill.pytorch import initialize_storage_
from shadowspill.training import DataParallel
from shadowspill.training.backends import ShadowSpill
from shadowspill.training.schedules import WarmupCosine
from workloads.models.glm53_flash.checkpoint import GLMCheckpoint
from workloads.models.shared.glm.initialization import InitializationProgress
from workloads.recipes.text.glm import (
    GLMTokenUpdates,
    fixed_batch_learning_report,
    next_token_prediction,
)
from workloads.recipes.text.profiling import NsightTrainer
from workloads.recipes.text.tokenizer import tokenizer_identity


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frozen-state-id", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--sequence-length", type=int, default=32768)
    parser.add_argument("--sequences-per-step", type=int, default=1, help="Per rank")
    parser.add_argument("--max-sequences-per-microbatch", type=int, default=1)
    parser.add_argument(
        "--steps", type=int, default=5, help="Total steps, including restored progress"
    )
    parser.add_argument("--schedule-steps", type=int, default=10)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=float, default=32)
    parser.add_argument(
        "--router-weight-dtype",
        choices=["bf16", "fp32"],
        help="Override checkpoint router weight storage (checkpoint default: BF16)",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--head-chunk-size", type=int, default=1024)
    parser.add_argument(
        "--layer-limit", type=int, help="Debug prefix; omit for full model"
    )
    parser.add_argument("--execution-gib", type=float, default=76)
    parser.add_argument("--spill-gib", type=float, default=96)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--eval-updates", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=2)
    parser.add_argument("--search-workers", type=int, default=8)
    parser.add_argument(
        "--resolution-options",
        nargs="+",
        default=["1"],
        help="Fractions of stages to recompute; default: 1 (all stages)",
    )
    parser.add_argument(
        "--nsys-steps",
        type=int,
        default=0,
        help="Capture this many updates under nsys cudaProfilerApi (default: disabled)",
    )
    parser.add_argument("--nsys-start-step", type=int, default=2)
    parser.add_argument(
        "--repeat-training-batch",
        action="store_true",
        help="Repeat each rank's first real-text window to check learning",
    )
    parser.add_argument(
        "--minimum-relative-loss-drop",
        type=float,
        help="Require a fixed-batch loss decrease, e.g. 0.01 for one percent",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if min(args.steps, args.lora_rank, args.head_chunk_size) < 1:
        parser.error("steps, LoRA rank and head chunk size must be positive")
    if args.schedule_steps < args.steps or args.schedule_steps <= 1:
        parser.error("schedule-steps must exceed one and cover steps")
    if args.nsys_steps < 0 or args.nsys_start_step < 1:
        parser.error("nsys-steps must be nonnegative and nsys-start-step positive")
    if args.nsys_steps:
        last = args.nsys_start_step + args.nsys_steps - 1
        if last > args.steps:
            parser.error("steps must cover the requested Nsight capture")
        for cadence in (args.eval_every, args.checkpoint_every):
            if cadence and any(
                step % cadence == 0 for step in range(args.nsys_start_step, last)
            ):
                parser.error("evaluation/checkpoint cadence must lie outside capture")
    if args.minimum_relative_loss_drop is not None:
        if not args.repeat_training_batch:
            parser.error("a loss-drop requirement needs --repeat-training-batch")
        if not 0 <= args.minimum_relative_loss_drop < 1 or args.steps < 4:
            parser.error("loss-drop must lie in [0, 1), with at least four steps")
    return args


def main(args):
    parallel = (
        DataParallel(shard_weights=True, shard_gradients=True, timeout=7200)
        if int(os.environ.get("WORLD_SIZE", "1")) > 1
        else None
    )
    rank = 0 if parallel is None else parallel.rank
    world_size = 1 if parallel is None else parallel.world_size
    meta = json.loads((args.data / "meta.json").read_text())
    identity = tokenizer_identity(args.checkpoint)
    if meta.get("tokenizer_sha256") != identity:
        raise ValueError("corpus tokenizer identity differs from this checkpoint")
    geometry = dict(
        sequence_length=args.sequence_length,
        sequences_per_step=args.sequences_per_step,
        max_sequences_per_microbatch=args.max_sequences_per_microbatch,
        rank=rank,
        world_size=world_size,
    )
    data = GLMTokenUpdates(
        args.data, repeat_first_update=args.repeat_training_batch, **geometry
    )
    if len(data) < args.steps:
        raise ValueError("corpus does not contain enough complete global updates")

    def validation_data():
        return GLMTokenUpdates(args.data, split="val", **geometry)

    source = GLMCheckpoint(args.checkpoint, expert_storage="fp8_block")
    model = source.build_model(
        layer_limit=args.layer_limit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_factor_dtype=torch.bfloat16,
        router_weight_dtype={"bf16": torch.bfloat16, "fp32": torch.float32}.get(
            args.router_weight_dtype
        ),
    )
    if meta["vocab_size"] > model.config.vocab_size:
        raise ValueError("corpus vocabulary exceeds model embeddings")
    if args.eval_every and len(validation_data()) < args.eval_updates:
        raise ValueError("validation corpus is shorter than requested evaluation")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (args.run_dir / "configuration.json").write_text(
            json.dumps(
                {
                    **{
                        key: str(value) if isinstance(value, Path) else value
                        for key, value in vars(args).items()
                    },
                    "world_size": world_size,
                    "inventory": source.inventory,
                    "tokenizer_sha256": identity,
                    "dataset": meta,
                    "shard_weights": parallel is not None,
                    "shard_gradients": parallel is not None,
                    "shard_optimizer": True,
                },
                indent=2,
            )
            + "\n"
        )
    frozen_id = (
        args.frozen_state_id
        + ":"
        + json.dumps(
            {
                "architecture": "glm53_flash",
                "layers": model.config.num_hidden_layers,
                "storage": source.inventory["storage_dtype_bytes"],
                "expert_storage": model.config.expert_storage,
                "dense_gemm": "bf16",
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
            },
            sort_keys=True,
        )
    )
    started = time.monotonic()

    def initialize(imported):
        progress = (
            InitializationProgress(imported, mode="checkpoint") if rank == 0 else None
        )

        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(args.seed)
                source.load_into(
                    imported, write_tensor=initialize_storage_, on_tensor=progress
                )
        finally:
            source.close()

    resolution_options = tuple(Fraction(value) for value in args.resolution_options)

    backend = ShadowSpill(
        execution_gib=args.execution_gib,
        spill_gib=args.spill_gib,
        artifact_store=args.run_dir / "artifacts" / f"rank-{rank:05d}",
        preparation_timeout=7200,
        external_headroom_gib=2,
        search_options=SearchOptions(
            workers=args.search_workers,
            algorithm=PressureFit(
                PressureFitOptions(resolution_options=resolution_options)
            ),
            generic=GenericPlanningOptions(
                minimum_object_bytes_evict_eligible=128 << 10
            ),
        ),
    )
    candidate_counts = {
        name: len(list(split(data.example())))
        for name, split in data.candidates.items()
    }

    def metrics(observations):
        # Objectives are already divided by the GLOBAL token count. Their rank
        # contributions sum to a token mean; averaging across ranks is incorrect.
        count = candidate_counts[trainer.selected_candidate]
        updates = len(observations.losses) / count
        total = torch.tensor(sum(observations.losses) / updates, dtype=torch.float64)
        if parallel is not None:
            dist.all_reduce(total)  # Gloo CPU control group, after the step sync.
        loss = total.item()
        if not math.isfinite(loss):
            raise FloatingPointError("nonfinite global loss")
        return {"global_mean_loss": loss}

    trainer = NsightTrainer(
        model,
        nsys_start_step=args.nsys_start_step,
        nsys_steps=args.nsys_steps,
        objective=partial(next_token_prediction, head_chunk_size=args.head_chunk_size),
        optimizer=AdamW,
        optimizer_args={"opt_state_dtype": torch.float32},
        schedules={
            "lr": WarmupCosine(
                lr=args.learning_rate,
                min_lr=args.learning_rate / 10,
                warmup_steps=1,
                total_steps=args.schedule_steps,
            )
        },
        master_dtype=torch.float32,
        grad_dtype=torch.float32,
        microbatch_candidates=data.candidates,
        backend=backend,
        metric_reducer=metrics,
        distributed=parallel,
        shard_optimizer=True,
        frozen_state_id=frozen_id,
    )

    training_losses = []

    def check_step(_trainer, result):
        if not math.isfinite(result.loss):
            raise FloatingPointError(f"nonfinite training loss at step {result.step}")
        training_losses.append(result.summary.scalars["global_mean_loss"])

    try:
        trainer.prepare(data.example(), initialize=initialize, checkpoint=args.resume)
        print(f"rank {rank}: prepared in {time.monotonic() - started:.1f}s", flush=True)
        trainer.fit(
            data,
            steps=args.steps,
            run_dir=args.run_dir,
            log_every=1,
            logger=lambda record: print(json.dumps(record), flush=True),
            callbacks=(check_step,),
            eval_data=validation_data if args.eval_every else None,
            eval_every=args.eval_every,
            eval_updates=args.eval_updates,
            checkpoint_every=args.checkpoint_every,
            keep_last=2,
        )
        if args.repeat_training_batch and len(training_losses) >= 4:
            learning = fixed_batch_learning_report(
                training_losses,
                minimum_relative_drop=args.minimum_relative_loss_drop or 0,
            )
            if rank == 0:
                (args.run_dir / "learning.json").write_text(
                    json.dumps(learning, indent=2) + "\n"
                )
                print("fixed-batch learning: " + json.dumps(learning), flush=True)
            if args.minimum_relative_loss_drop is not None and not learning["passed"]:
                raise RuntimeError(
                    "fixed-batch loss did not meet the requested decrease; "
                    "inspect learning.json before further benchmarks"
                )
        elif args.minimum_relative_loss_drop is not None:
            raise ValueError("loss-drop check needs four newly executed updates")
        if rank == 0:
            (args.run_dir / "completed.json").write_text(
                json.dumps(
                    {
                        "steps": trainer.step_count,
                        "next_data_step": data.next_step,
                        "elapsed_seconds": time.monotonic() - started,
                    },
                    indent=2,
                )
                + "\n"
            )
    finally:
        trainer.close()
        source.close()


if __name__ == "__main__":
    main(arguments())
