"""Direct Executor checks for captured workload fixtures; no public training loop."""

import json

from shadowspill import Executor, Plan, load_checkpoint, save_checkpoint
from shadowspill.checkpoint import metadata_values
from shadowspill.orchestration.launch.workers import workers
from shadowspill.pytorch._rng import restore_rng, rng_state
from shadowspill.training.observations import StepObservations
from shadowspill.training.preparation.configuration import adapter
from shadowspill.training.preparation.phases import settings


def run(directory):
    root, options = settings(directory)
    options.update(json.loads((root / "selected.json").read_text()))
    metadata = json.loads((root / "model.json").read_text())
    definition = adapter(options["recipe"])
    runtime = dict(definition.trainer_options(options, metadata))
    schedules = runtime.pop("schedules", {})
    reducer = runtime.pop("metric_reducer", None)
    frozen = runtime.pop("frozen_state_id", None)
    norms_every = runtime.pop("parameter_norms_every", 0)
    evaluation = (
        {"evaluation": Plan.load(root / "plans/evaluation.ssp")}
        if options["eval_every"]
        else {}
    )
    records = []
    with (
        workers(options["ranks"]) as group,
        Executor(
            Plan.load(root / "plans/training.ssp"),
            plans=evaluation,
            rank_map=group.rank_map,
            parameter_norms=bool(norms_every),
            **runtime,
        ) as executor,
    ):
        source = definition.inputs(options, rank=group.logical_rank)
        with definition.initializer(options, metadata, rank=group.logical_rank) as fill:
            if options["resume"]:
                saved = load_checkpoint(
                    executor,
                    options["resume"].format(rank=group.logical_rank),
                    fill_tensor=fill,
                    frozen_state_id=frozen,
                )
                position = saved["step"]
                source.load_state_dict(metadata_values.decode(saved["data"]))
                restore_rng(metadata_values.decode(saved["rng"]), executor.device)
            else:
                executor.initialize(fill)
                position = 0
        training = root / "training"
        local = (
            training / f"rank-{group.logical_rank:05d}" if group.count > 1 else training
        )
        local.mkdir(parents=True, exist_ok=True)
        checkpoints = local / "checkpoints"
        while position < options["steps"]:
            controls = {
                name: schedule(position) for name, schedule in schedules.items()
            }
            raw = executor.run(
                **next(source),
                hyperparams=controls,
                collect_parameter_norms=bool(
                    norms_every and (position + 1) % norms_every == 0
                ),
            ).result()
            observed = StepObservations.collect(
                raw.losses, raw.metrics, raw.parameter_metrics
            )
            executor.synchronize()
            del raw
            position += 1
            scalars = reducer(observed) if reducer is not None else {}
            loss = sum(observed.losses)
            records.append({"step": position, "loss": loss, "scalars": scalars})
            with (local / "metrics.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "step": position,
                            "train/loss": loss,
                            **{f"train/{key}": value for key, value in scalars.items()},
                        }
                    )
                    + "\n"
                )
            if options["eval_every"] and (
                position % options["eval_every"] == 0 or position == options["steps"]
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
                        total += float(
                            executor.run(
                                plan="evaluation",
                                args=[args],
                                kwargs=[kwargs],
                            ).result()
                        )
                value = definition.evaluation_metric(total / options["eval_updates"])
                if group.rank == 0:
                    with (training / "evaluation.jsonl").open("a") as stream:
                        stream.write(
                            json.dumps(
                                {
                                    "step": position,
                                    "validation_loss": value,
                                }
                            )
                            + "\n"
                        )
            if options["checkpoint_every"] and (
                position % options["checkpoint_every"] == 0
                or position == options["steps"]
            ):
                checkpoints.mkdir(exist_ok=True)
                save_checkpoint(
                    executor,
                    checkpoints / f"step_{position:08d}.ssp",
                    frozen_state_id=frozen,
                    metadata={
                        "step": position,
                        "data": metadata_values.encode(source.state_dict()),
                        "rng": metadata_values.encode(rng_state(executor.device)),
                    },
                )
                for path in sorted(checkpoints.glob("step_*.ssp"))[
                    : -options["keep_last"]
                ]:
                    path.unlink()
        finish = getattr(definition, "finish", None)
        if finish is not None:
            finish(root, options, records, rank=group.logical_rank)
        per_rank = group.gather(
            {
                "rank": group.logical_rank,
                "steps": records,
                "spill_capacity_bytes": executor.spill_capacity_bytes,
            }
        )
        if group.rank == 0:
            (root / "completed.json").write_text(
                json.dumps(
                    {
                        "steps": position,
                        "data_state": source.state_dict(),
                        "per_rank": per_rank,
                    },
                    indent=2,
                )
                + "\n"
            )


if __name__ == "__main__":
    import sys

    run(sys.argv[1])
