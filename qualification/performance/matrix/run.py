"""Run the selected performance cells and aggregate their verdicts."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from typing import TYPE_CHECKING

from qualification.device_defaults import performance_defaults
from qualification.matrix_logging import MatrixConsole, utc_now
from qualification.performance.cases import manifests
from qualification.precision import dtype_overrides
from shadowspill.schema import artifact_schema

from .arguments import create_parser, default_cells, parse_cell_budgets
from .case import run_cell
from .report import protocol

if TYPE_CHECKING:
    from shadowspill.network import RemotePool


def main(spill: RemotePool | None = None) -> int:
    """Run the same cells with host storage or the supplied remote pool."""

    parser = create_parser()
    arguments = parser.parse_args()
    performance_defaults(arguments)
    if (
        arguments.execution_budget_gib is not None
        and arguments.execution_budget_gib <= 0
    ):
        parser.error("--execution-budget-gib must be positive")
    if spill is not None:
        # Travels to each cell as a string, so a cell run by hand is the same
        # cell the matrix spawned.
        arguments.remote_spill = f"{spill.host}:{spill.port}:{spill.capacity}"
    if arguments.checkpoint and arguments.plan_only:
        parser.error("--checkpoint has no effect with --plan-only")
    if arguments.measure_only and arguments.plan_only:
        parser.error("--measure-only has nothing to measure with --plan-only")
    identities = frozenset(manifest.identity for manifest in manifests())
    try:
        planning_budgets = parse_cell_budgets(
            arguments.planning_spill_budget_gib, identities
        )
        pool_budgets = parse_cell_budgets(
            arguments.spill_budget_gib, identities, flag="spill-budget-gib"
        )
    except ValueError as error:
        parser.error(str(error))
    output = arguments.output_directory.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected = set(arguments.cells or ())
    if selected:
        chosen = [item for item in manifests() if item.identity in selected]
    else:
        chosen = list(default_cells())
    # A smaller pool is the cell's manifest from here on, so what the banner
    # and each cell's record say is the pool it actually ran with.
    chosen = [
        replace(
            manifest,
            spill_budget_bytes=pool_budgets.get(
                manifest.identity, manifest.spill_budget_bytes >> 30
            )
            << 30,
            device_physical_capacity_bytes=(
                (arguments.execution_budget_gib << 30)
                if arguments.execution_budget_gib is not None
                else manifest.device_physical_capacity_bytes
            ),
            external_headroom_bytes=arguments.external_headroom_mib << 20,
            reject_overbudget=arguments.reject_overbudget,
            **dtype_overrides(arguments),
        )
        for manifest in chosen
    ]
    rows: list[dict[str, object]] = []
    failed = False
    matrix_started = time.perf_counter()
    with MatrixConsole(output / "matrix.log") as console:
        console.block(
            "MATRIX START",
            [
                f"UTC: {utc_now()}",
                f"OUTPUT: {output}",
                f"CELLS: {len(chosen)} of {len(manifests())}: "
                + ", ".join(manifest.identity for manifest in chosen),
                "PROTOCOL: "
                + protocol(
                    plan_only=arguments.plan_only,
                    checkpoint=arguments.checkpoint,
                    measure_only=arguments.measure_only,
                ),
            ],
        )
        for ordinal, manifest in enumerate(chosen, start=1):
            row = run_cell(
                manifest,
                arguments,
                output=output,
                planning_budgets=planning_budgets,
                console=console,
                prefix=f"[{ordinal}/{len(chosen)}]",
            )
            rows.append(row)
            if row["return_code"] != 0 or not row["passed"]:
                failed = True
                if not arguments.keep_going:
                    break
        summary = {
            "schema": artifact_schema("full_model_matrix"),
            "plan_only": arguments.plan_only,
            "measure_only": arguments.measure_only,
            "cells": rows,
            "passed": bool(rows) and not failed and all(row["passed"] for row in rows),
        }
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        console.emit()
        if arguments.measure_only:
            banner = "MATRIX MEASURED" if summary["passed"] else "MATRIX ERROR"
            counted = "CELLS MEASURED: "
        else:
            banner = "MATRIX " + ("PASS" if summary["passed"] else "FAIL")
            counted = "CELLS PASSED: "
        console.block(
            banner,
            [
                counted + f"{sum(1 for row in rows if row['passed'])}/{len(chosen)}",
                f"SUMMARY: {output / 'summary.json'}",
                f"STOP: {utc_now()}",
                f"DURATION: {time.perf_counter() - matrix_started:.3f} seconds",
            ],
        )
    return 0 if summary["passed"] else 1
