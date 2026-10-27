"""Numerical matrix verdicts, summaries and console details."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from qualification.matrix_logging import MatrixConsole, format_bytes, utc_now
from qualification.numerical.references import (
    canonical_reference_path,
    reference_artifact_exists,
)
from qualification.precision import dtype_description
from shadowspill.schema import artifact_schema

from .commands import CaseOptions


@dataclass(frozen=True, slots=True)
class CaseResult:
    family: str
    implementation: str
    device_budget_bytes: int
    elapsed_seconds: float
    return_code: int
    reference: str
    artifact: str
    passed: bool
    failure_categories: tuple[str, ...] = ()
    failures: tuple[str, ...] = ()


def case_verdict(
    artifact: Path, return_code: int
) -> tuple[bool, int, tuple[str, ...], tuple[str, ...]]:
    """Read what the case says about itself, and say so when it said nothing.

    A case that judges itself failed exits non-zero, having already written
    the artifact saying why, so the artifact is read whenever it exists
    rather than only on a clean exit.
    """

    payload: dict[str, object] | None = None
    if artifact.is_file():
        try:
            candidate = json.loads(artifact.read_text())
        except json.JSONDecodeError:
            candidate = None
        if isinstance(candidate, dict) and candidate.get("schema") == artifact_schema(
            "numerical_qualification"
        ):
            payload = candidate
    passed = False
    failure_categories: tuple[str, ...] = ()
    failures: tuple[str, ...] = ()
    if payload is not None:
        passed = bool(return_code == 0 and payload.get("passed") is True)
        # Read back from JSON, so every member is object until it is checked.
        raw_categories = payload.get("failure_categories")
        failure_categories = (
            tuple(str(item) for item in raw_categories)
            if isinstance(raw_categories, list)
            else ()
        )
        raw_failures = payload.get("failures")
        failures = (
            tuple(
                f"{item['category']}: {item['detail']}"
                for item in raw_failures
                if isinstance(item, dict)
            )
            if isinstance(raw_failures, list)
            else ()
        )
    if not passed and return_code == 0:
        return_code = 1
    if not passed and not failures:
        # The case never got far enough to judge itself, which is its own kind
        # of failure and must not be read as a numerical disagreement.
        failure_categories = ("process",)
        failures = (f"process: exited {return_code} without a usable artifact",)
    return passed, return_code, failure_categories, failures


def matrix_summary(
    results: list[CaseResult],
    selected_cases: list[tuple[str, str]],
    *,
    empty_caches: bool,
) -> dict[str, object]:
    """The matrix artifact: one record per case, and whether all of them passed."""

    return {
        "schema": artifact_schema("model_correctness_matrix"),
        "passed": len(results) == len(selected_cases)
        and all(item.passed for item in results),
        "empty_caches": empty_caches,
        "cases": [
            {
                "family": item.family,
                "implementation": item.implementation,
                "device_budget_bytes": item.device_budget_bytes,
                "elapsed_seconds": item.elapsed_seconds,
                "return_code": item.return_code,
                "reference": item.reference,
                "artifact": item.artifact,
                "passed": item.passed,
                "failure_categories": list(item.failure_categories),
                "failures": list(item.failures),
            }
            for item in results
        ],
    }


def announce_case(
    console: MatrixConsole,
    options: CaseOptions,
    *,
    family: str,
    implementation: str,
    budget: int,
    progress: str,
    case_log: Path,
) -> str:
    """Say what is about to run, and return the time it started."""

    reference = canonical_reference_path(
        options.reference_directory,
        model_name=family,
        implementation=implementation,
    )
    reference_state = (
        "regenerating"
        if options.regenerate_reference or not reference_artifact_exists(reference)
        else "reusing canonical"
    )
    started_at = utc_now()
    console.emit()
    console.block(
        f"CASE START {progress} {implementation}_{family}",
        [
            f"MODEL: {implementation}/{family}",
            f"DEVICE BUDGET: {format_bytes(budget)}",
            "EXTERNAL HEADROOM: "
            + (
                "512"
                if options.external_headroom_mib is None
                else str(options.external_headroom_mib)
            )
            + " MiB (reserved for pool sizing)",
            f"REJECT OVERBUDGET: {options.reject_overbudget}",
            dtype_description(options),
            f"MODEL CONFIG: {options.model_config}",
            f"REFERENCE: {reference} ({reference_state})",
            f"LOG: {case_log}",
            f"START: {started_at}",
        ],
    )
    return started_at
