"""Launch the ShadowSpill-only full-model qualification cells.

The gate runs the three mlops cells by default. Those are the ones that
carry a throughput authority, so they are the only ones that can pass or
fail; the pure-PyTorch variants have no regression floor to compare
against and only cost wall time. `--cells` still reaches any of them,
including the PyTorch ones, when a run wants them.
"""

from __future__ import annotations

import argparse
import json
import re
import signal
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from workloads.full_model import FullModelManifest, manifests

from .matrix_logging import MatrixConsole, format_bytes, utc_now

# Cell logs may carry one leading "[<utc>] " stamp added by the matrix tee.
_PLAN_PROGRESS = re.compile(
    r"^(?:\[[^\]]*\] )?\[shadowspill\.plan \+\s*[0-9.]+s\]\s+"
    r"(?P<phase>[A-Za-z0-9_]+): "
    r"(?P<state>started|finished|failed)(?:\s+in\s+.*)?$"
)
_PLAN_LINE = re.compile(r"^(?:\[[^\]]*\] )?(?P<line>\[shadowspill\.plan .*)$")


def _active_planning_phases(log_text: str) -> tuple[str, ...]:
    """Recover the open phase stack from verbose planning output."""

    active: list[str] = []
    for line in log_text.splitlines():
        match = _PLAN_PROGRESS.match(line)
        if match is None:
            continue
        phase = match.group("phase")
        if match.group("state") == "started":
            active.append(phase)
            continue
        for index in range(len(active) - 1, -1, -1):
            if active[index] == phase:
                del active[index:]
                break
    return tuple(active)


def _termination_signal(return_code: int) -> str | None:
    if return_code >= 0:
        return None
    try:
        return signal.Signals(-return_code).name
    except ValueError:
        return f"signal_{-return_code}"


def _write_parent_failure(
    *,
    manifest_identity: str,
    return_code: int,
    log: Path,
    failure_path: Path,
) -> dict[str, object]:
    """Record failures a killed child had no opportunity to serialize."""

    text = log.read_text(errors="replace") if log.is_file() else ""
    active = _active_planning_phases(text)
    signal_name = _termination_signal(return_code)
    phase = active[-1] if active else None
    if signal_name is not None:
        message = f"qualification subprocess terminated by {signal_name}"
    else:
        message = f"qualification subprocess exited with status {return_code}"
    if phase is not None:
        message += f" during planning phase {phase!r}"
    message += "; no Python traceback was available"
    progress_lines = [
        match.group("line")
        for match in map(_PLAN_LINE.match, text.splitlines())
        if match is not None
    ]
    failure: dict[str, object] = {
        "schema": "shadowspill.full_model_subprocess_failure/v1",
        "identity": manifest_identity,
        "error_type": "SubprocessTermination",
        "error": message,
        "return_code": return_code,
        "termination_signal": signal_name,
        "active_planning_phases": active,
        "last_planning_progress": progress_lines[-1] if progress_lines else None,
        "log": str(log),
    }
    failure_path.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
    return failure


def _parse_cell_planning_budgets(
    entries: Sequence[str],
    identities: frozenset[str],
) -> dict[str, int]:
    """Parse repeatable ``IDENTITY=GIB`` planning-budget overrides."""

    budgets: dict[str, int] = {}
    for entry in entries:
        identity, separator, gib_text = entry.partition("=")
        if not separator:
            raise ValueError(
                f"planning-spill-budget-gib entry {entry!r} must be IDENTITY=GIB"
            )
        if identity not in identities:
            raise ValueError(
                f"planning-spill-budget-gib names unknown cell {identity!r}"
            )
        if identity in budgets:
            raise ValueError(
                f"planning-spill-budget-gib repeats cell {identity!r}"
            )
        gib = int(gib_text)
        if gib <= 0:
            raise ValueError(
                f"planning-spill-budget-gib for {identity!r} must be positive"
            )
        budgets[identity] = gib
    return budgets


def _cell_start_details(
    manifest: FullModelManifest,
    *,
    planning_budget_gib: int | None,
    checkpoint: bool,
    plan_only: bool,
    log: Path,
    started_at: str,
) -> list[str]:
    """Describe one cell exactly as it is about to run."""

    if plan_only:
        protocol = "plan only"
    elif checkpoint:
        protocol = "checkpoint, warm step, restore, 3x4 measured steps"
    else:
        protocol = "throughput probe without checkpoint, warm step, 3x4 measured steps"
    details = [
        f"MODEL: {manifest.implementation}/{manifest.family}",
        "DATA GEOMETRY:",
        f"  SEQUENCE LENGTH: {manifest.sequence_length} tokens",
        f"  TOKENS PER MICROBATCH: {manifest.tokens_per_microbatch}",
        f"  SEQUENCES PER MICROBATCH: {manifest.sequences_per_microbatch}",
        f"  GRADIENT ACCUMULATION ROUNDS: {manifest.accumulation_count}",
        f"  TOKENS PER OPTIMIZER STEP: {manifest.tokens_per_step}",
        "EXECUTION BUDGET: "
        + format_bytes(manifest.device_physical_capacity_bytes),
        f"SPILL BUDGET: {format_bytes(manifest.spill_budget_bytes)}",
    ]
    if planning_budget_gib is not None:
        details.append(
            f"PLANNING SPILL BUDGET: {format_bytes(planning_budget_gib << 30)}"
        )
    details.extend(
        (
            f"PROTOCOL: {protocol}",
            f"LOG: {log}",
            f"START: {started_at}",
        )
    )
    return details


def _cell_result_details(
    artifact_payload: dict[str, object] | None,
    failure: dict[str, object] | None,
    *,
    started_at: str,
    elapsed: float,
) -> list[str]:
    """Summarize one finished cell's gates, evidence, and timing."""

    details: list[str] = []
    if artifact_payload is not None:
        if not artifact_payload.get("plan_only"):
            gates = (
                ("protocol_complete", "PROTOCOL"),
                ("objectives_finite", "OBJECTIVES"),
                ("logical_steps_passed", "LOGICAL STEPS"),
                ("physical_budget_passed", "PHYSICAL BUDGETS"),
                ("strict_runtime_passed", "STRICT RUNTIME"),
                ("simulator_gate_passed", "SIMULATOR"),
                ("regression_gate_passed", "REGRESSION"),
            )
            for key, label in gates:
                value = artifact_payload.get(key)
                details.append(f"GATE {label}: {'pass' if value else 'FAIL'}")
            median_step = artifact_payload.get("median_step_seconds")
            throughput = artifact_payload.get("median_tokens_per_second")
            if isinstance(median_step, float) and isinstance(throughput, float):
                details.append(
                    f"MEDIAN STEP: {median_step:.4f} seconds "
                    f"({throughput:.1f} tokens/s)"
                )
            error = artifact_payload.get("simulator_relative_error")
            predicted = artifact_payload.get("predicted_makespan_seconds")
            if isinstance(predicted, float) and isinstance(error, float):
                details.append(
                    f"PREDICTED STEP: {predicted:.4f} seconds "
                    f"(simulator error {error:+.2%})"
                )
            ratio = artifact_payload.get("regression_throughput_ratio")
            if isinstance(ratio, float):
                details.append(f"REGRESSION RATIO: {ratio:.2%}")
            ratio = artifact_payload.get("predecessor_throughput_ratio")
            if isinstance(ratio, float):
                details.append(f"PREDECESSOR RATIO: {ratio:.2%}")
        planning = artifact_payload.get("planning_seconds")
        if isinstance(planning, float):
            details.append(f"PLANNING: {planning:.3f} seconds")
    if failure is not None:
        details.append(f"ERROR TYPE: {failure.get('error_type')}")
        details.append(f"ERROR: {failure.get('error')}")
    details.extend(
        (
            f"START: {started_at}",
            f"STOP: {utc_now()}",
            f"DURATION: {elapsed:.3f} seconds",
        )
    )
    return details


def default_cells() -> tuple[FullModelManifest, ...]:
    """The cells the gate runs when none are named.

    A cell without a throughput authority has no floor to be measured
    against, so running it can neither pass nor fail -- it only spends wall
    time. The default is therefore the judgeable set, and `--cells` still
    reaches the rest.
    """

    return tuple(
        item for item in manifests() if item.regression_tokens_per_second is not None
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("qualification/results/full_model"),
    )
    parser.add_argument("--force-fresh", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--checkpoint",
        action="store_true",
        help=(
            "opt into the anonymous full-state checkpoint/restore protocol; "
            "the matrix default is a throughput probe without that copy"
        ),
    )
    parser.add_argument("--implementation-revision")
    parser.add_argument(
        "--cells",
        nargs="*",
        help=(
            "identities to run, such as mlops_llama3 or pytorch_qwen35; "
            "defaults to every cell carrying a throughput authority"
        ),
    )
    parser.add_argument(
        "--planning-spill-budget-gib",
        action="append",
        default=[],
        metavar="IDENTITY=GIB",
        help=(
            "per-cell planning spill budget within the configured pool, "
            "for example mlops_qwen35=100; repeatable"
        ),
    )
    arguments = parser.parse_args()
    if arguments.checkpoint and arguments.plan_only:
        parser.error("--checkpoint has no effect with --plan-only")
    identities = frozenset(manifest.identity for manifest in manifests())
    try:
        planning_budgets = _parse_cell_planning_budgets(
            arguments.planning_spill_budget_gib, identities
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
    if arguments.plan_only:
        mode = "plan only"
    elif arguments.checkpoint:
        mode = "checkpoint, warm step, restore, 3x4 measured steps"
    else:
        mode = "throughput probe without checkpoint, warm step, 3x4 measured steps"
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
                f"PROTOCOL: {mode}",
            ],
        )
        for ordinal, manifest in enumerate(chosen, start=1):
            prefix = f"[{ordinal}/{len(chosen)}]"
            artifact = output / f"{manifest.identity}.json"
            failure_path = artifact.with_suffix(".failure.json")
            log = output / f"{manifest.identity}.log"
            # A rerun must never be classified from artifacts of an older run.
            artifact.unlink(missing_ok=True)
            failure_path.unlink(missing_ok=True)
            log.unlink(missing_ok=True)
            command = [
                sys.executable,
                "-m",
                "tools.qualification.performance",
                manifest.family,
                manifest.implementation,
                str(artifact),
                "--artifact-store-dir",
                str(output / "artifact_store" / manifest.identity),
            ]
            if arguments.force_fresh:
                command.append("--force-fresh")
            if arguments.plan_only:
                command.append("--plan-only")
            elif not arguments.checkpoint:
                # The matrix default is a checkpoint-free throughput probe: the
                # anonymous full-state copy cannot coexist with the full pinned
                # spill arena on qualification hosts.  Checkpoint/replay
                # coverage stays in the numerical matrix and behind
                # --checkpoint here.
                command.append("--skip-checkpoint")
            if manifest.identity in planning_budgets:
                command.extend(
                    (
                        "--planning-spill-budget-gib",
                        str(planning_budgets[manifest.identity]),
                    )
                )
            if arguments.implementation_revision is not None:
                command.extend(
                    ("--implementation-revision", arguments.implementation_revision)
                )
            started = time.perf_counter()
            started_at = utc_now()
            console.emit()
            console.block(
                f"CELL START {prefix} {manifest.identity}",
                _cell_start_details(
                    manifest,
                    planning_budget_gib=planning_budgets.get(manifest.identity),
                    checkpoint=arguments.checkpoint,
                    plan_only=arguments.plan_only,
                    log=log,
                    started_at=started_at,
                ),
            )
            return_code = console.stream(command, cell_log_path=log, prefix=prefix)
            elapsed = time.perf_counter() - started
            artifact_payload: dict[str, object] | None = None
            if artifact.is_file():
                artifact_payload = json.loads(artifact.read_text())
            passed = bool(
                artifact_payload.get("passed") if artifact_payload else False
            )
            failure_record: dict[str, object] | None = None
            if failure_path.is_file():
                failure_record = json.loads(failure_path.read_text())
            elif return_code != 0 and artifact_payload is None:
                failure_record = _write_parent_failure(
                    manifest_identity=manifest.identity,
                    return_code=return_code,
                    log=log,
                    failure_path=failure_path,
                )
            status = "PASS" if return_code == 0 and passed else "FAIL"
            console.block(
                f"CELL {status} {prefix} {manifest.identity}",
                _cell_result_details(
                    artifact_payload,
                    failure_record,
                    started_at=started_at,
                    elapsed=elapsed,
                ),
            )
            row = {
                "identity": manifest.identity,
                "return_code": return_code,
                "passed": passed,
                "elapsed_seconds": elapsed,
                "artifact": str(artifact),
                "artifact_exists": artifact.is_file(),
                "failure_artifact": (
                    str(failure_path) if failure_record is not None else None
                ),
                "failure": failure_record,
                "log": str(log),
            }
            rows.append(row)
            if return_code != 0 or not passed:
                failed = True
                if not arguments.keep_going:
                    break
        summary = {
            "schema": "shadowspill.full_model_matrix/v1",
            "plan_only": arguments.plan_only,
            "cells": rows,
            "passed": bool(rows)
            and not failed
            and all(row["passed"] for row in rows),
        }
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        console.emit()
        console.block(
            "MATRIX " + ("PASS" if summary["passed"] else "FAIL"),
            [
                "CELLS PASSED: "
                f"{sum(1 for row in rows if row['passed'])}/{len(chosen)}",
                f"SUMMARY: {output / 'summary.json'}",
                f"STOP: {utc_now()}",
                f"DURATION: {time.perf_counter() - matrix_started:.3f} seconds",
            ],
        )
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
