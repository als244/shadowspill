"""Measured budget comparisons from completed runs; no model or GPU imports."""

import csv

from .phases.common import read, write

FIELDS = (
    "budget",
    "candidate",
    "predicted_seconds",
    "median_seconds",
    "units_per_step",
    "unit_label",
    "units_per_second",
)


def write_measurements(root, winners, *, plots):
    destination = root / "execution"
    rows = []
    for winner in winners:
        path = destination / winner["file"].removesuffix(".ssp") / "result.json"
        if not path.exists():
            continue
        value = read(path)
        if any(
            value[key] != winner[key] for key in ("candidate", "budget", "plan_digest")
        ):
            continue  # Replanning can replace the Plan at the same path.
        rows.append({key: value[key] for key in FIELDS})
    rows.sort(key=lambda row: row["budget"])
    write(destination / "summary.json", rows)
    with (destination / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    for extension in ("svg", "png"):
        (destination / f"performance.{extension}").unlink(missing_ok=True)
    if plots and rows:
        _plot(rows, destination)


def _plot(rows, destination):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    throughput = all(row["units_per_second"] is not None for row in rows)
    figure = Figure(figsize=(12 if throughput else 6, 4), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2 if throughput else 1, squeeze=False)[0]
    budgets = [row["budget"] / 2**30 for row in rows]
    axes[0].plot(
        budgets, [row["median_seconds"] for row in rows], "o-", label="Measured median"
    )
    axes[0].plot(
        budgets, [row["predicted_seconds"] for row in rows], "s--", label="Simulated"
    )
    axes[0].set_ylabel("Time per update (s)")
    axes[0].legend()
    if throughput:
        axes[1].plot(budgets, [row["units_per_second"] for row in rows], "o-")
        axes[1].set_ylabel(f"Throughput ({rows[0]['unit_label']}/s)")
    for axis in axes:
        axis.set_xlabel("Total GPU budget (GiB)")
        axis.grid(alpha=0.2)
    figure.suptitle("Quickstart execution")
    figure.savefig(destination / "performance.svg")
    figure.savefig(destination / "performance.png", dpi=150)
    figure.clear()
