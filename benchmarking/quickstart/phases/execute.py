"""One budget in a fresh worker; never builds, measures, or replans."""

import statistics

from shadowspill import Plan, Trainer

from .common import experiment, paths, plan_identity, read, write


def run(request, selection):
    root, index = paths(request)
    winner = read(root / "plans" / "index.json")[selection]
    plan = Plan.load(root / "plans" / winner["file"])
    if plan_identity(plan) != winner["plan_digest"]:
        raise ValueError("selected Plan differs from the planning index")
    directory = root / "execution" / winner["file"].removesuffix(".ssp")
    directory.mkdir(parents=True, exist_ok=True)
    attempt = directory / f"run-{len(list(directory.glob('run-*'))):04d}"
    attempt.mkdir()
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
    with Trainer(
        plan, device=device, external_headroom_bytes=request["external_headroom_bytes"]
    ) as trainer:
        setup = experiment(request)
        trainer.metric_reducer = setup.get("metric_reducer")
        trainer.initialize(setup["fill_tensor"])
        inputs = dict(setup["candidates"][winner["candidate"]])
        inputs["hyperparams"] = setup.get("hyperparams", {})
        trainer.fit(
            (inputs for _ in range(request["steps"])),
            steps=request["steps"],
            run_dir=attempt,
            callbacks=[record],
        )
    median = statistics.median(record["seconds"] for record in records)
    units = index.get("units_per_step")
    write(
        directory / "result.json",
        {
            **winner,
            "steps": records,
            "run_directory": str(attempt.relative_to(root)),
            "median_seconds": median,
            "units_per_step": units,
            "unit_label": index["unit_label"],
            "units_per_second": None if units is None else units / median,
        },
    )
