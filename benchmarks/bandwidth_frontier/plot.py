"""Draw persisted numeric results; no build, profile or planning imports."""

import json
from collections import defaultdict

from matplotlib.figure import Figure


def draw(source, destination, *, title=None):
    root = source if (source / "frontier.json").exists() else source / "raw_data"
    request = json.loads((root / "sweep.json").read_text())
    rows = json.loads((root / "frontier.json").read_text())
    observed = root / "measured.json"
    measured = json.loads(observed.read_text()) if observed.exists() else []
    figures = destination / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for spill in request["spill_gib"]:
        curves = defaultdict(dict)
        for row in rows:
            if row["spill_budget_gib"] == spill and row["is_winner"]:
                key = row["fetch_gb_per_second"], row["evict_gb_per_second"]
                curves[key][row["execution_budget_gib"]] = row["units_per_second"]
        figure = Figure(figsize=(8, 5), layout="constrained")
        axes = figure.subplots()
        for (fetch, evict), points in curves.items():
            budgets = sorted(set(request["budgets_gib"]))
            axes.plot(
                budgets,
                [points.get(b, float("nan")) for b in budgets],
                marker="o",
                label=f"{fetch:g}/{evict:g} GB/s",
            )
        for entry in measured:
            if entry.get("spill_gib") not in (None, spill):
                continue
            points = sorted(
                (float(b), float(v)) for b, v in entry["units_per_second"].items()
            )
            axes.plot(
                *zip(*points, strict=True),
                marker="X",
                linestyle="--",
                label=entry.get("label", "measured"),
            )
        axes.set(
            xlabel="GPU memory budget (GiB)",
            ylabel=f"Throughput ({request['unit_label']}/s)",
            title=f"{title or request['title']} — {spill:g} GiB spill",
            ylim=(0, None),
        )
        axes.grid(axis="y", alpha=0.25)
        if curves or measured:
            axes.legend(title="Fetch/evict; simulated lines")
        figure.savefig(figures / f"throughput-spill-{spill:g}.png", dpi=150)
