"""Run traces for pool-occupancy diagnostics."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..intervals import GIB
from ..stored import StoredPlan, _budget_label


def _traced_steps(
    run_root: Path, plans: Sequence[StoredPlan]
) -> tuple[dict[Path, Mapping[str, Any]], list[tuple[str, str]]]:
    """Each traced step's diagnostics, keyed by the plan it ran, found by
    the makespan the step records at the budget its file is named for; and
    the steps no stored plan matches."""

    matched: dict[Path, Mapping[str, Any]] = {}
    missing: list[tuple[str, str]] = []
    for trace in sorted((run_root / "steps").glob("*gib.json")):
        diagnostics = json.loads(trace.read_text())
        makespan_ns = round(
            float(diagnostics["summary"]["simulator_makespan_seconds"]) * 1e9
        )
        budget_bytes = round(float(trace.stem.removesuffix("gib")) * GIB)
        candidates = [
            record
            for record in plans
            if not record.resolution and abs(record.makespan_ns - makespan_ns) <= 1_000
        ]
        # One plan can be a candidate at several budgets, at one makespan and a
        # larger pool at each, so the makespan alone does not say which ran.
        # The runner names the file for its budget as `_budget_label` labels a
        # plan's, to six digits; without that label, the budget ran the largest
        # pool that fits under it, and never one it could not hold.
        labelled = [
            record
            for record in candidates
            if record.budget_bytes is not None
            and _budget_label(record.budget_bytes) == trace.stem
        ]
        fitting = [
            record
            for record in candidates
            if record.pool_capacity_bytes <= budget_bytes
        ]
        chosen = (
            labelled[0]
            if labelled
            else max(
                fitting, key=lambda record: record.pool_capacity_bytes, default=None
            )
        )
        if chosen is not None:
            matched[chosen.path] = diagnostics
        else:
            missing.append(
                (
                    trace.stem,
                    (
                        "no stored plan within this budget has"
                        if candidates
                        else "no stored plan has"
                    )
                    + f" this step's simulated makespan ({makespan_ns / 1e9:.3f} s)",
                )
            )
    return matched, missing
