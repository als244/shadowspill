"""Execute saved Plans with bounded initialization and stateful real inputs."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from shadowspill.training.results import StepResult
    from shadowspill.training.trainer import Trainer
import json
import os
import sys
import time
from contextlib import ExitStack
from pathlib import Path

from shadowspill.orchestration.monitoring.lifecycle import Lifecycle


def run(directory: str | Path) -> None:
    rank = int(os.environ.get("RANK", "0"))
    path = Path(directory) / "logs" / f"execution-rank-{rank:05d}.jsonl"
    with Lifecycle(path, rank=rank) as timing, timing.phase("execution"):
        _execute(directory, timing)


def _execute(directory: str | Path, timing: Lifecycle) -> None:
    with timing.phase("imports"):
        from shadowspill import Plan, Trainer
        from shadowspill.orchestration.launch.workers import workers
        from shadowspill.training.logging import DistributedLogger, Wandb
        from shadowspill.training.preparation.phases import settings

        from .configuration import adapter

    with timing.phase("configuration"):
        root, options = settings(directory)
        options.update(json.loads((root / "selected.json").read_text()))
        metadata = json.loads((root / "model.json").read_text())
        definition = adapter(options["recipe"])
    started = time.monotonic()
    with (
        timing.managed("control_group", workers(options["ranks"])) as group,
        ExitStack() as stack,
    ):
        with timing.phase("plans.load"):
            plan = Plan.load(root / "plans/training.ssp")
            evaluation = (
                Plan.load(root / "plans/evaluation.ssp")
                if options["eval_every"]
                else None
            )
        with timing.phase("runtime.create"):
            trainer_type = Trainer
            trainer_options = dict(definition.trainer_options(options, metadata))
            if options.get("nsys_steps", 0):
                from shadowspill.orchestration.capture.nsys import nsight_trainer

                trainer_type = nsight_trainer(trainer_type)
                trainer_options.update(
                    nsys_start_step=options["nsys_start_step"],
                    nsys_steps=options["nsys_steps"],
                )
            trainer = trainer_type(
                plan,
                evaluation_plan=evaluation,
                rank_map=group.rank_map,
                **trainer_options,
            )
            trainer = stack.enter_context(timing.managed("runtime", trainer))
        # The execution allocator is installed before an adapter may load weights.
        with (
            timing.phase("weights.initialize"),
            definition.initializer(
                options, metadata, rank=group.logical_rank
            ) as initialize,
        ):
            if options["resume"]:
                trainer.load(
                    options["resume"].format(rank=group.logical_rank),
                    fill_tensor=initialize,
                )
            else:
                trainer.initialize(initialize)
        print(
            f"rank {group.logical_rank}: initialized in "
            f"{time.monotonic() - started:.1f}s; "
            f"spill pool {trainer.executor.spill_capacity_bytes / (1 << 30):.3f} GiB",
            flush=True,
        )
        with timing.phase("inputs.create"):
            source = definition.inputs(options, rank=group.logical_rank)
        training = root / "training"
        training.mkdir(exist_ok=True)
        local = (
            training / f"rank-{group.logical_rank:05d}" if group.count > 1 else training
        )
        with timing.phase("logging.setup"):
            logger: Wandb | DistributedLogger | None = None
            if group.count > 1:
                import torch.distributed as dist

                logger = stack.enter_context(
                    timing.managed(
                        "logging",
                        DistributedLogger(
                            dist.group.WORLD,
                            run_dir=training,
                            wandb={
                                "project": options["wandb_project"],
                                "config": options,
                            }
                            if options["wandb_project"]
                            else None,
                        ),
                    )
                )
            elif options["wandb_project"]:
                logger = stack.enter_context(
                    timing.managed(
                        "logging",
                        Wandb(
                            project=options["wandb_project"],
                            run_dir=training,
                            config=options,
                        ),
                    )
                )
        stream = (
            stack.enter_context((training / "evaluation.jsonl").open("a", buffering=1))
            if group.rank == 0 and options["eval_every"]
            else None
        )
        records: list[dict[str, Any]] = []

        def completed(owner: "Trainer", result: "StepResult") -> None:
            # This callback follows the existing scalar/table logging. It is a
            # host progress marker; result.seconds is Trainer's update timer.
            timing.event(
                "training.step_completed", step=result.step, seconds=result.seconds
            )
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

        with timing.phase("training.fit"):
            trainer.fit(
                source,
                steps=options["steps"],
                run_dir=local,
                logger=logger,
                callbacks=[completed],
                checkpoint_every=options["checkpoint_every"],
                keep_last=options["keep_last"],
            )
        with timing.phase("adapter.finish"):
            finish = getattr(definition, "finish", None)
            if finish is not None:
                finish(root, options, records, rank=group.logical_rank)
        with timing.phase("results.gather"):
            per_rank = group.gather(
                dict(
                    rank=group.logical_rank,
                    steps=records,
                    spill_capacity_bytes=trainer.executor.spill_capacity_bytes,
                )
            )
        with timing.phase("results.write"):
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
