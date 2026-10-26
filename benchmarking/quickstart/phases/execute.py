"""One budget in a fresh worker; never builds, measures, or replans."""

import statistics
from contextlib import nullcontext

from shadowspill import Plan, Trainer

from .common import experiment, paths, plan_identity, read, write
from .distributed import workers


def run(request, selection):
    root, index = paths(request)
    with workers(request, index) as group:
        _execute(request, selection, root, index, group)


def _execute(request, selection, root, index, group):
    winner = read(root / "plans" / "index.json")[selection]
    plan = Plan.load(root / "plans" / winner["file"])
    if plan_identity(plan) != winner["plan_digest"]:
        raise ValueError("selected Plan differs from the planning index")
    directory = root / "execution" / winner["file"].removesuffix(".ssp")
    attempt_name = None
    if group.rank == 0:
        directory.mkdir(parents=True, exist_ok=True)
        attempt_name = f"run-{len(list(directory.glob('run-*'))):04d}"
        (directory / attempt_name).mkdir()
    attempt = directory / group.gather(attempt_name)[0]
    run_dir = (
        attempt if group.count == 1 else attempt / f"rank-{group.logical_rank:04d}"
    )
    records = []

    def record(owner, result):
        records.append(
            {
                "step": result.step,
                "seconds": result.seconds,
                "loss": float(result.summary.scalars.get("loss", result.loss)),
                "objective_loss": result.loss,
            }
        )

    # Install Executor before the user factory can create live CUDA resources.
    device = int(request["device"].split(":")[-1])
    with Trainer(plan, device=device, rank_map=group.rank_map) as trainer:
        setup = experiment(
            request, logical_rank=None if group.count == 1 else group.logical_rank
        )
        trainer.metric_reducer = setup.get("metric_reducer")
        with setup.get("context", nullcontext)():
            trainer.initialize(setup["fill_tensor"])
        inputs = dict(setup["candidates"][winner["candidate"]])
        inputs["hyperparams"] = setup.get("hyperparams", {})
        trainer.fit(
            (inputs for _ in range(request["steps"])),
            steps=request["steps"],
            run_dir=run_dir,
            callbacks=[record],
        )
    per_rank = group.gather({"logical_rank": group.logical_rank, "steps": records})
    if group.rank != 0:
        return
    per_rank.sort(key=lambda row: row["logical_rank"])
    records = aggregate_steps(per_rank)
    median = statistics.median(record["seconds"] for record in records)
    units = index.get("units_per_step")
    write(
        directory / "result.json",
        {
            **winner,
            "steps": records,
            "per_rank": per_rank,
            "run_directory": str(attempt.relative_to(root)),
            "median_seconds": median,
            "units_per_step": units,
            "unit_label": index["unit_label"],
            "units_per_second": None if units is None else units / median,
        },
    )


def aggregate_steps(per_rank):
    """Report the slowest worker and sum normalized objective contributions."""
    combined = []
    for rows in zip(*(rank["steps"] for rank in per_rank), strict=True):
        if len({row["step"] for row in rows}) != 1:
            raise ValueError("workers reported different training steps")
        combined.append(
            {
                "step": rows[0]["step"],
                "seconds": max(row["seconds"] for row in rows),
                "loss": sum(row["loss"] for row in rows),
                "objective_loss": sum(row["objective_loss"] for row in rows),
            }
        )
    return combined
