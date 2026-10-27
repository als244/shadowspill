"""Record experiment runs and merge their completed results."""

from __future__ import annotations

import datetime
import json
from pathlib import Path


def now() -> str:
    return datetime.datetime.now().astimezone().isoformat()


def run_stamp(output: Path) -> str:
    """A new run's name: its local start time, unique among the runs."""

    base = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stamp, number = base, 1
    while any(
        (output / "runs" / f"{stamp}{suffix}").exists() for suffix in (".json", ".log")
    ):
        number += 1
        stamp = f"{base}_{number}"
    return stamp


def merged_record(output: Path, spec: dict, run: dict) -> dict:
    """``experiment.json``: the latest spec, each sweep's latest run, a
    sweep never run marked so, and every run of the experiment."""

    path = output / "experiment.json"
    previous = json.loads(path.read_text()) if path.is_file() else {}
    held = {sweep["name"]: sweep for sweep in previous.get("sweeps", [])}
    for sweep in spec["sweeps"]:
        held.setdefault(
            sweep["name"],
            {
                **sweep,
                "output": str(output / "sweeps" / sweep["name"]),
                "status": "not run",
            },
        )
    for entry in run["sweeps"]:
        held[entry["name"]] = {**entry, "run": run["run"]}
    names = [sweep["name"] for sweep in spec["sweeps"]]
    runs = [item for item in previous.get("runs", []) if item.get("run") != run["run"]]
    runs.append(
        {
            "run": run["run"],
            "started_at": run["started_at"],
            "finished_at": run["finished_at"],
            "dry_run": run["dry_run"],
            "selected": run["selected"],
            "record": f"runs/{run['run']}.json",
            "console": f"runs/{run['run']}.log",
        }
    )
    return {
        "schema_version": 1,
        "spec_path": run["spec_path"],
        "spec": spec,
        "sweeps": [held[name] for name in names]
        + [entry for name, entry in held.items() if name not in names],
        "runs": runs,
        "report": run["report"]
        if run["report"] is not None
        else previous.get("report"),
        "summary": (
            run["summary"] if run["summary"] is not None else previous.get("summary")
        ),
    }
