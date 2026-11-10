"""Execute saved Plans with bounded initialization and stateful real inputs."""

import json
import sys
import time
from contextlib import ExitStack

from benchmarking.quickstart.phases.distributed import workers
from shadowspill import Plan, Trainer
from shadowspill.training.logging import DistributedLogger, Wandb
from workloads.recipes.phases import settings

from .configuration import adapter


def run(directory):
    root, options = settings(directory)
    options.update(json.loads((root / "selected.json").read_text()))
    metadata = json.loads((root / "model.json").read_text())
    definition = adapter(options["recipe"])
    started = time.monotonic()
    with workers({}, {"world_size": options["ranks"]}) as group, ExitStack() as stack:
        trainer_type = getattr(definition, "trainer_type", lambda: Trainer)()
        trainer = stack.enter_context(
            trainer_type(
                Plan.load(root / "plans/training.ssp"),
                evaluation_plan=Plan.load(root / "plans/evaluation.ssp")
                if options["eval_every"]
                else None,
                rank_map=group.rank_map,
                **definition.trainer_options(options, metadata),
            )
        )
        # The execution allocator is installed before an adapter may load weights.
        with definition.initializer(
            options, metadata, rank=group.logical_rank
        ) as initialize:
            if options["resume"]:
                trainer.load(
                    options["resume"].format(rank=group.logical_rank),
                    fill_tensor=initialize,
                )
            else:
                trainer.initialize(initialize)
        print(
            f"rank {group.logical_rank}: initialized in "
            f"{time.monotonic() - started:.1f}s",
            flush=True,
        )
        source = definition.inputs(options, rank=group.logical_rank)
        training = root / "training"
        training.mkdir(exist_ok=True)
        local = (
            training / f"rank-{group.logical_rank:05d}" if group.count > 1 else training
        )
        logger = None
        if group.count > 1:
            import torch.distributed as dist

            logger = stack.enter_context(
                DistributedLogger(
                    dist.group.WORLD,
                    run_dir=training,
                    wandb={"project": options["wandb_project"], "config": options}
                    if options["wandb_project"]
                    else None,
                )
            )
        elif options["wandb_project"]:
            logger = stack.enter_context(
                Wandb(
                    project=options["wandb_project"],
                    run_dir=training,
                    config=options,
                )
            )
        stream = (
            stack.enter_context((training / "evaluation.jsonl").open("a", buffering=1))
            if group.rank == 0 and options["eval_every"]
            else None
        )
        records = []

        def completed(owner, result):
            records.append(
                dict(
                    step=result.step,
                    seconds=result.seconds,
                    loss=result.loss,
                    scalars=dict(result.summary.scalars),
                )
            )
            if options["eval_every"] and (
                result.step % options["eval_every"] == 0
                or result.step == options["steps"]
            ):
                validation = definition.inputs(
                    options, rank=group.logical_rank, split="val"
                )
                total = 0.0
                for _ in range(options["eval_updates"]):
                    calls = next(validation)
                    for args, kwargs in zip(
                        calls["args"], calls["kwargs"], strict=True
                    ):
                        total += float(owner.evaluate(args=[args], kwargs=[kwargs]))
                value = definition.evaluation_metric(total / options["eval_updates"])
                if stream is not None:
                    record = dict(step=result.step, validation_loss=value)
                    stream.write(json.dumps(record) + "\n")
                    print(record, flush=True)

        trainer.fit(
            source,
            steps=options["steps"],
            run_dir=local,
            logger=logger,
            callbacks=[completed],
            checkpoint_every=options["checkpoint_every"],
            keep_last=options["keep_last"],
        )
        finish = getattr(definition, "finish", None)
        if finish is not None:
            finish(root, options, records, rank=group.logical_rank)
        per_rank = group.gather(dict(rank=group.logical_rank, steps=records))
        if group.rank == 0:
            (root / "completed.json").write_text(
                json.dumps(
                    dict(
                        steps=trainer.step_count,
                        elapsed_seconds=time.monotonic() - started,
                        data_state=source.state_dict(),
                        per_rank=per_rank,
                    ),
                    indent=2,
                )
                + "\n"
            )


if __name__ == "__main__":
    run(sys.argv[1])
