"""Fresh workers load saved Plans and fill bounded checkpoint shards."""

import json
import sys
import time
from contextlib import ExitStack
from datetime import timedelta

import torch.distributed as dist

from shadowspill import Plan
from shadowspill.training.schedules import WarmupCosine
from workloads.recipes.phases import settings
from workloads.recipes.text.profiling import NsightTrainer

from .inputs import Inputs
from .metrics import evaluate, learning_report, training_metrics
from .model import checkpoint, describe


def run(directory):
    root, options = settings(directory)
    options.update(json.loads((root / "selected.json").read_text()))
    metadata = json.loads((root / "model.json").read_text())
    started, ranks = time.monotonic(), options["ranks"]
    with ExitStack() as stack:
        if ranks > 1:
            dist.init_process_group("gloo", timeout=timedelta(seconds=7200))
            stack.callback(dist.destroy_process_group)
        rank = dist.get_rank() if ranks > 1 else 0
        training_root = root / "training"
        training_root.mkdir(exist_ok=True)
        local = training_root / f"rank-{rank:05d}" if ranks > 1 else training_root
        schedule = WarmupCosine(
            lr=options["learning_rate"],
            min_lr=options["learning_rate"] / 10,
            warmup_steps=1,
            total_steps=options["schedule_steps"],
        )
        trainer = stack.enter_context(
            NsightTrainer(
                Plan.load(root / "plans/training.ssp"),
                evaluation_plan=Plan.load(root / "plans/evaluation.ssp")
                if options["eval_every"]
                else None,
                rank_map={i: i for i in range(ranks)},
                schedules={"optimizer_group.0.lr": schedule},
                frozen_state_id=metadata["frozen_state_id"],
                metric_reducer=training_metrics,
                nsys_start_step=options["nsys_start_step"],
                nsys_steps=options["nsys_steps"],
            )
        )
        with checkpoint(options) as (model, source, initialize):
            if describe(options, model, source) != metadata:
                raise ValueError("checkpoint/corpus metadata differs from the build")
            if options["resume"]:
                trainer.load(
                    options["resume"].format(rank=rank), fill_tensor=initialize
                )
            else:
                trainer.initialize(initialize)
        print(
            f"rank {rank}: initialized in {time.monotonic() - started:.1f}s", flush=True
        )
        data = Inputs(options, rank=rank, world_size=ranks)
        losses = []
        stream = (
            stack.enter_context(
                (training_root / "evaluation.jsonl").open("a", buffering=1)
            )
            if rank == 0
            else None
        )

        def completed(owner, result):
            losses.append(result.summary.scalars["global_mean_loss"])
            if options["eval_every"] and (
                result.step % options["eval_every"] == 0
                or result.step == options["steps"]
            ):
                evaluate(owner, options, rank=rank, stream=stream)

        trainer.fit(
            data,
            steps=options["steps"],
            run_dir=local,
            callbacks=[completed],
            checkpoint_every=options["checkpoint_every"],
            keep_last=2,
        )
        learning_report(root, options, losses, rank)
        if rank == 0:
            (root / "completed.json").write_text(
                json.dumps(
                    {
                        "steps": trainer.step_count,
                        "next_data_step": data.data.next_step,
                        "elapsed_seconds": time.monotonic() - started,
                    },
                    indent=2,
                )
                + "\n"
            )


if __name__ == "__main__":
    run(sys.argv[1])
