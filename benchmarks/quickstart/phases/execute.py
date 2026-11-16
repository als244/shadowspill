"""One budget in a fresh worker; never builds, measures, or replans."""

import statistics
import time
from contextlib import nullcontext
from itertools import repeat

from benchmarks.plan_runner.updates import updates
from benchmarks.spill import resolve as spill_pool
from shadowspill import Executor, Plan
from shadowspill.orchestration.launch.workers import workers
from shadowspill.training._logs import emit

from .candidates import example_inputs
from .common import experiment, paths, plan_identity, read, write


def run(request, selection):
    root, index = paths(request)
    with workers(index.get("world_size", 1), rank_map=request.get("rank_map")) as group:
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

    # Install Executor before the user factory can create live CUDA resources.
    device = int(request["device"].split(":")[-1])
    with Executor(
        plan, device=device, rank_map=group.rank_map, spill=spill_pool(request)
    ) as executor:
        started = time.perf_counter()
        setup = experiment(
            request, logical_rank=None if group.count == 1 else group.logical_rank
        )
        with setup.get("context", nullcontext)():
            executor.initialize(setup["fill_tensor"])
        inputs = example_inputs(setup["candidates"][winner["candidate"]])
        inputs["hyperparams"] = setup.get("hyperparams", {})
        run_dir.mkdir(exist_ok=True)
        with (run_dir / "metrics.jsonl").open("x", buffering=1) as stream:
            for result in updates(
                executor,
                repeat(inputs, request["steps"]),
                steps=request["steps"],
                schedules=None,
                metric_reducer=setup.get("metric_reducer"),
                parameter_norms_every=0,
                nsys=None,
                rank=group.rank,
            ):
                emit(result, time.perf_counter() - started, stream, None)
                records.append(
                    {
                        "step": result.step,
                        "seconds": result.seconds,
                        "loss": float(result.summary.scalars.get("loss", result.loss)),
                        "objective_loss": result.loss,
                    }
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
