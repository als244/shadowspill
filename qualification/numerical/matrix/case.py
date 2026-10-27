"""Execute warm-up, reference and saved-phase workers for one case."""

from __future__ import annotations

import time
from pathlib import Path

from qualification.matrix_logging import MatrixConsole
from qualification.numerical.references import canonical_reference_path
from qualification.pipeline.cache import compiler_cache

from .commands import CaseOptions, case_commands
from .report import CaseResult, case_verdict


def _run_commands(
    commands: list[list[str]],
    *,
    prefix: str,
    options: CaseOptions,
    console: MatrixConsole,
    progress: str,
    case_log: Path,
) -> int:
    """Run each arm in its own process, in the caches the options ask for."""

    return_code = 0
    for command in commands:
        mode = command[command.index("qualification.numerical.run") + 1]
        is_warm = mode == "_warm"
        is_reference = mode == "_reference"
        if not (is_warm or is_reference) and options.detailed_artifacts:
            command.append("--detailed-artifacts")
        phase = (
            "kernel warm-up: compile and tune, record nothing"
            if is_warm
            else "compiled reference generation"
            if is_reference
            else "planned parity, checkpoint replay, and physical budgets"
        )
        console.emit(f"PHASE: {phase}", prefix=progress)
        with compiler_cache(
            options.environment,
            empty=options.empty_caches,
            parent=options.cache_directory,
            prefix=f"{prefix}-",
        ) as environment:
            return_code = console.stream(
                command,
                cell_log_path=case_log,
                prefix=progress,
                environment=environment,
            )
        if return_code != 0:
            break
    return return_code


def run_case(
    *,
    family: str,
    implementation: str,
    device_budget: int,
    output_directory: Path,
    options: CaseOptions,
    console: MatrixConsole,
    progress: str,
    case_log: Path,
) -> CaseResult:
    """Run one cell of the matrix, and read the verdict it wrote."""

    prefix = f"{implementation}_{family}"
    reference = canonical_reference_path(
        options.reference_directory,
        model_name=family,
        implementation=implementation,
    )
    artifact = output_directory / f"{prefix}.json"
    started = time.perf_counter()
    return_code = _run_commands(
        case_commands(
            family, implementation, device_budget, reference, artifact, options
        ),
        prefix=prefix,
        options=options,
        console=console,
        progress=progress,
        case_log=case_log,
    )
    passed, return_code, failure_categories, failures = case_verdict(
        artifact, return_code
    )
    return CaseResult(
        family=family,
        implementation=implementation,
        device_budget_bytes=device_budget,
        elapsed_seconds=time.perf_counter() - started,
        return_code=return_code,
        reference=str(reference),
        artifact=str(artifact),
        passed=passed,
        failure_categories=failure_categories,
        failures=failures,
    )
