"""Execute a performance cell and retain its verdict or failure."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from qualification.matrix_logging import MatrixConsole, utc_now
from qualification.performance.cases import FullModelManifest
from qualification.pipeline.cache import compiler_cache

from .command import cell_command
from .failures import write_parent_failure
from .report import cell_result_details, cell_start_details


def run_cell(
    manifest: FullModelManifest,
    arguments: argparse.Namespace,
    *,
    output: Path,
    planning_budgets: dict[str, int],
    console: MatrixConsole,
    prefix: str,
) -> dict[str, object]:
    """Run one cell, and record what it wrote or failed to write."""

    artifact = output / f"{manifest.identity}.json"
    failure_path = artifact.with_suffix(".failure.json")
    log = output / f"{manifest.identity}.log"
    # A rerun must never be classified from artifacts of an older run.
    artifact.unlink(missing_ok=True)
    failure_path.unlink(missing_ok=True)
    log.unlink(missing_ok=True)
    command = cell_command(
        manifest,
        arguments,
        artifact,
        planning_budgets,
    )
    started = time.perf_counter()
    started_at = utc_now()
    console.emit()
    console.block(
        f"CELL START {prefix} {manifest.identity}",
        cell_start_details(
            manifest,
            planning_budget_gib=planning_budgets.get(manifest.identity),
            checkpoint=arguments.checkpoint,
            plan_only=arguments.plan_only,
            log=log,
            started_at=started_at,
        ),
    )
    with compiler_cache(
        empty=arguments.empty_caches, prefix=f"shadowspill-{manifest.identity}-"
    ) as environment:
        return_code = console.stream(
            command, cell_log_path=log, prefix=prefix, environment=environment
        )
    elapsed = time.perf_counter() - started
    artifact_payload: dict[str, object] | None = None
    if artifact.is_file():
        artifact_payload = json.loads(artifact.read_text())
    # Measure-only asks whether the cell ran, not whether it was good
    # enough; the cell subprocess already declines to fail on gates.
    passed = bool(
        (artifact_payload is not None)
        if arguments.measure_only
        else (artifact_payload.get("passed") if artifact_payload else False)
    )
    failure_record: dict[str, object] | None = None
    if failure_path.is_file():
        failure_record = json.loads(failure_path.read_text())
    elif return_code != 0 and artifact_payload is None:
        failure_record = write_parent_failure(
            manifest_identity=manifest.identity,
            return_code=return_code,
            log=log,
            failure_path=failure_path,
        )
    ran = return_code == 0 and passed
    if arguments.measure_only:
        status = "MEASURED" if ran else "ERROR"
    else:
        status = "PASS" if ran else "FAIL"
    console.block(
        f"CELL {status} {prefix} {manifest.identity}",
        cell_result_details(
            artifact_payload,
            failure_record,
            started_at=started_at,
            elapsed=elapsed,
            measure_only=arguments.measure_only,
        ),
    )
    return {
        "identity": manifest.identity,
        "return_code": return_code,
        "passed": passed,
        "elapsed_seconds": elapsed,
        "artifact": str(artifact),
        "artifact_exists": artifact.is_file(),
        "failure_artifact": (str(failure_path) if failure_record is not None else None),
        "failure": failure_record,
        "log": str(log),
    }
