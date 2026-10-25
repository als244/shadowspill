"""Numeric results are sufficient to redraw figures without loading a Plan."""

import csv
import json
import math

GIB = 1 << 30


def rows(result, fetch, evict, units):
    winners = result.best_per_budget
    for case in result.cases:
        summary = {} if case.plan is None else case.plan.summary
        duration = summary.get("duration_seconds")
        yield {
            "fetch_gb_per_second": fetch,
            "evict_gb_per_second": evict,
            "execution_budget_gib": case.memory_budgets.execution_bytes / GIB,
            "spill_budget_gib": case.memory_budgets.spill_bytes / GIB,
            "candidate": case.candidate,
            "seconds": duration,
            "units_per_second": None if not duration else units / duration,
            "is_winner": case.plan is not None
            and winners[case.memory_budgets] is case.plan,
            "status": "planned" if case.plan is not None else case.failure["type"],
            "error": case.failure.get("message", ""),
            "fetch_bytes": summary.get("fetch_bytes"),
            "evict_bytes": summary.get("evict_bytes"),
            "spill_peak_bytes": summary.get("spill_peak_bytes"),
            "simulation_only": summary.get("simulation_only"),
        }


def write_tables(root, records):
    (root / "frontier.json").write_text(json.dumps(records, indent=2) + "\n")
    with (root / "frontier.csv").open("w", newline="") as stream:
        if records:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


def collect_measured(source, root):
    """Accept explicit observations, including the old tokens/s spelling."""
    value = json.loads(source.read_text())
    entries = value if isinstance(value, list) else value["measured"]
    records = []
    for entry in entries:
        points = entry.get("units_per_second", entry.get("tokens_per_second", {}))
        if not points or any(
            not math.isfinite(float(v)) or float(v) <= 0
            for point in points.items()
            for v in point
        ):
            raise ValueError(
                "measured points require positive finite budgets and throughput"
            )
        records.append({**entry, "units_per_second": points})
    (root / "measured.json").write_text(json.dumps(records, indent=2) + "\n")
