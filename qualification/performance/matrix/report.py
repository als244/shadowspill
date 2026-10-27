"""Console descriptions of performance cases, timing and gate outcomes."""

from __future__ import annotations

from pathlib import Path

from qualification.matrix_logging import format_bytes, utc_now
from qualification.performance.cases import FullModelManifest


def cell_start_details(
    manifest: FullModelManifest,
    *,
    planning_budget_gib: int | None,
    checkpoint: bool,
    plan_only: bool,
    log: Path,
    started_at: str,
) -> list[str]:
    """Describe one cell exactly as it is about to run."""

    details = [
        f"MODEL: {manifest.implementation}/{manifest.family}",
        "DATA GEOMETRY:",
        f"  SEQUENCE LENGTH: {manifest.sequence_length} tokens",
        f"  TOKENS PER MICROBATCH: {manifest.tokens_per_microbatch}",
        f"  SEQUENCES PER MICROBATCH: {manifest.sequences_per_microbatch}",
        f"  GRADIENT ACCUMULATION ROUNDS: {manifest.accumulation_count}",
        f"  TOKENS PER OPTIMIZER STEP: {manifest.tokens_per_step}",
        "EXECUTION BUDGET: " + format_bytes(manifest.device_physical_capacity_bytes),
        "EXTERNAL HEADROOM: "
        + format_bytes(manifest.external_headroom_bytes)
        + " (reserved for pool sizing)",
        f"REJECT OVERBUDGET: {manifest.reject_overbudget}",
        "DTYPES: " + manifest.dtypes.description(),
        f"SPILL BUDGET: {format_bytes(manifest.spill_budget_bytes)}",
    ]
    if planning_budget_gib is not None:
        details.append(
            f"PLANNING SPILL BUDGET: {format_bytes(planning_budget_gib << 30)}"
        )
    details.extend(
        (
            "PROTOCOL: " + protocol(plan_only=plan_only, checkpoint=checkpoint),
            f"LOG: {log}",
            f"START: {started_at}",
        )
    )
    return details


def cell_result_details(
    artifact_payload: dict[str, object] | None,
    failure: dict[str, object] | None,
    *,
    started_at: str,
    elapsed: float,
    measure_only: bool = False,
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
            if not measure_only:
                for key, label in gates:
                    value = artifact_payload.get(key)
                    if (
                        key == "regression_gate_passed"
                        and artifact_payload.get("regression_gate_applicable") is False
                    ):
                        details.append(
                            "GATE REGRESSION: not applicable ("
                            + str(artifact_payload["regression_comparison"]["reason"])
                            + ")"
                        )
                    else:
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
            # Both ratios divide by throughput measured on the machine that
            # set the floors, so on another machine they describe the
            # hardware rather than this run.
            if not measure_only:
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


def protocol(*, plan_only: bool, checkpoint: bool, measure_only: bool = False) -> str:
    """Name the protocol the cells will run, for the matrix banner."""

    if plan_only:
        mode = "plan only"
    elif checkpoint:
        mode = "checkpoint, warm step, restore, 3x4 measured steps"
    else:
        mode = "throughput probe without checkpoint, warm step, 3x4 measured steps"
    if measure_only:
        mode += "; reported without gates"
    return mode
