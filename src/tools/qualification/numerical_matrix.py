"""Launch the model-scale numerical and physical-budget qualification matrix."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from workloads.numerical import DEFAULT_DEVICE_BUDGETS

from .matrix_logging import MatrixConsole, format_bytes, utc_now
from .references import (
    DEFAULT_APPROXIMATELY_1B_REFERENCE_DIRECTORY,
    canonical_reference_path,
    reference_artifact_exists,
)

_FAMILIES: Final = tuple(DEFAULT_DEVICE_BUDGETS)
_IMPLEMENTATIONS: Final = ("pytorch", "mlops")
_DEFAULT_IMPLEMENTATIONS: Final = {
    "llama3": _IMPLEMENTATIONS,
    "qwen35": _IMPLEMENTATIONS,
    "olmoe": ("mlops",),
}


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


def _parse_bytes(value: str) -> int:
    normalized = value.strip().lower().replace("_", "")
    factors = (("gib", 1 << 30), ("mib", 1 << 20), ("kib", 1 << 10))
    for suffix, factor in factors:
        if normalized.endswith(suffix):
            number = normalized[: -len(suffix)]
            if not number.isdigit():
                raise argparse.ArgumentTypeError(f"invalid byte count {value!r}")
            result = int(number) * factor
            if result <= 0:
                raise argparse.ArgumentTypeError("byte count must be positive")
            return result
    if not normalized.isdigit() or int(normalized) <= 0:
        raise argparse.ArgumentTypeError(f"invalid positive byte count {value!r}")
    return int(normalized)


def _budget_overrides(
    values: list[str], *, valid_models: set[str] | None = None
) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        family, separator, budget = value.partition("=")
        if (
            separator == ""
            or not family
            or (valid_models is not None and family not in valid_models)
        ):
            raise argparse.ArgumentTypeError(
                "budget must be MODEL=BYTES for one of "
                + ", ".join(sorted(valid_models or ()))
            )
        result[family] = _parse_bytes(budget)
    return result


def _run_case(
    *,
    family: str,
    implementation: str,
    device_budget: int,
    output_directory: Path,
    reference_directory: Path,
    environment: dict[str, str],
    regenerate_reference: bool,
    seed: int,
    model_config: str,
    data_geometry: str | None,
    case_factory: str | None,
    case_options: list[str],
    optimizer_ordering: str,
    cold: bool,
    cache_directory: Path | None,
    detailed_artifacts: bool,
    console: MatrixConsole,
    progress: str,
    case_log: Path,
) -> CaseResult:
    prefix = f"{implementation}_{family}"
    reference = canonical_reference_path(
        reference_directory,
        model_name=family,
        implementation=implementation,
    )
    artifact = output_directory / f"{prefix}.json"
    base = [sys.executable, "-m", "tools.qualification.numerical"]
    options = [
        "--seed",
        str(seed),
        "--model-config",
        model_config,
        "--optimizer-ordering",
        optimizer_ordering,
    ]
    if data_geometry is not None:
        options.extend(("--data-geometry", data_geometry))
    if case_factory is not None:
        options.extend(("--case-factory", case_factory))
    for value in case_options:
        options.extend(("--case-option", value))
    commands: list[list[str]] = []
    if regenerate_reference or not reference_artifact_exists(reference):
        commands.append(
            [
                *base,
                "_reference",
                family,
                str(reference),
                "--model-implementation",
                implementation,
                *options,
            ]
        )
    commands.append(
        [
            *base,
            "_planned",
            family,
            str(reference),
            str(artifact),
            str(device_budget),
            "--model-implementation",
            implementation,
            *options,
        ]
    )
    started = time.perf_counter()
    return_code = 0
    for command_index, command in enumerate(commands):
        command_environment = dict(environment)
        is_reference = command_index == 0 and len(commands) == 2
        plan_cache = (
            (output_directory / "artifact_store" / prefix)
            if cache_directory is None
            else cache_directory.expanduser().resolve() / prefix
        )
        cache_root: Path | None = None
        if cold:
            cache_parent = (
                output_directory / ".cold_work"
                if cache_directory is None
                else cache_directory.expanduser().resolve()
            )
            cache_parent.mkdir(parents=True, exist_ok=True)
            cache_root = Path(
                tempfile.mkdtemp(prefix=f"{prefix}-", dir=cache_parent)
            ) / ("reference" if is_reference else "plan")
            cache_root.mkdir(parents=True, exist_ok=False)
            plan_cache = cache_root / "shadowspill"
            command_environment["TORCHINDUCTOR_CACHE_DIR"] = str(
                cache_root / "torchinductor"
            )
            command_environment["TRITON_CACHE_DIR"] = str(cache_root / "triton")
        if not is_reference:
            command.extend(("--artifact-store-dir", str(plan_cache)))
            if not detailed_artifacts:
                command.append("--no-save-plan")
            if detailed_artifacts:
                command.append("--detailed-artifacts")
            if cold:
                command.append("--force-fresh")
        phase = (
            "compiled reference generation"
            if is_reference
            else "planned parity, checkpoint replay, and physical budgets"
        )
        console.emit(f"PHASE: {phase}", prefix=progress)
        try:
            return_code = console.stream(
                command,
                cell_log_path=case_log,
                prefix=progress,
                environment=command_environment,
            )
        finally:
            if cache_root is not None:
                shutil.rmtree(cache_root.parent)
        if return_code != 0:
            break
    passed = False
    if return_code == 0 and artifact.is_file():
        payload = json.loads(artifact.read_text())
        passed = bool(
            payload.get("schema") == "shadowspill.numerical_qualification/v5"
            and payload.get("passed") is True
        )
        if not passed:
            return_code = 1
    return CaseResult(
        family=family,
        implementation=implementation,
        device_budget_bytes=device_budget,
        elapsed_seconds=time.perf_counter() - started,
        return_code=return_code,
        reference=str(reference),
        artifact=str(artifact),
        passed=passed,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run fresh-process compiled-reference/planned parity, checkpoint "
            "replay, transfer, and physical-budget gates while reporting "
            "recomputation selections diagnostically."
        )
    )
    parser.add_argument(
        "--models",
        "--families",
        dest="models",
        nargs="+",
        default=_FAMILIES,
        help="built-in family names or names consumed by --case-factory",
    )
    parser.add_argument(
        "--implementations",
        nargs="+",
        choices=_IMPLEMENTATIONS,
        help=(
            "explicit provider cross-product; omitted runs the supported "
            "five-cell matrix (pure-PyTorch OLMoE is deferred)"
        ),
    )
    parser.add_argument(
        "--budget",
        action="append",
        default=[],
        metavar="FAMILY=BYTES",
        help="override one family budget; suffixes KiB, MiB, and GiB are accepted",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("qualification/results/numerical_matrix"),
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=DEFAULT_APPROXIMATELY_1B_REFERENCE_DIRECTORY,
        help=(
            "canonical compiled-reference root; one identity-checked reference "
            "is retained under each model/provider directory"
        ),
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--cold",
        action="store_true",
        help=(
            "give every reference and planned subprocess fresh ShadowSpill, "
            "Inductor, and Triton cache roots"
        ),
    )
    parser.add_argument("--seed", type=int, default=20_260_811)
    parser.add_argument(
        "--optimizer-ordering",
        choices=("stage_interleaved", "tail"),
        default="stage_interleaved",
        help=(
            "group optimizer updates by stage and place them at their gradient "
            "frontier"
        ),
    )
    parser.add_argument(
        "--model-config",
        default="{}",
        metavar="JSON|@FILE",
        help="model configuration passed to every selected case",
    )
    parser.add_argument(
        "--data-geometry",
        metavar="JSON|@FILE",
        help="microbatch geometry passed to every selected case",
    )
    parser.add_argument(
        "--case-factory",
        metavar="MODULE:FUNCTION",
        help="qualification-case factory for custom model names",
    )
    parser.add_argument(
        "--case-option",
        action="append",
        default=[],
        metavar="NAME=JSON",
        help="repeatable custom-factory argument",
    )
    parser.add_argument(
        "--regenerate-reference",
        action="store_true",
        help="replace canonical references instead of reusing compatible files",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="continue remaining cases after a failed case",
    )
    parser.add_argument(
        "--detailed-artifacts",
        action="store_true",
        help=(
            "retain full PlanReports and per-task traces; compact correctness "
            "evidence and ephemeral cold caches are the default"
        ),
    )
    arguments = parser.parse_args()
    try:
        invalid_names = [
            name
            for name in arguments.models
            if re.fullmatch(r"[A-Za-z0-9_.-]+", name) is None
        ]
        if invalid_names:
            raise RuntimeError(
                "model names must contain only letters, digits, '.', '_', or '-': "
                + ", ".join(invalid_names)
            )
        custom_names = [name for name in arguments.models if name not in _FAMILIES]
        if custom_names and arguments.case_factory is None:
            raise RuntimeError(
                "custom model names require --case-factory MODULE:FUNCTION"
            )
        overrides = _budget_overrides(
            arguments.budget, valid_models=set(arguments.models)
        )
        missing_budgets = [name for name in custom_names if name not in overrides]
        if missing_budgets:
            raise RuntimeError(
                "custom model budgets must be explicit with --budget: "
                + ", ".join(missing_budgets)
            )
        environment = dict(os.environ)
    except (argparse.ArgumentTypeError, FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))

    output_directory = arguments.output_dir.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    reference_directory = arguments.reference_dir.expanduser().resolve()
    reference_directory.mkdir(parents=True, exist_ok=True)
    selected_cases = [
        (family, implementation)
        for family in arguments.models
        for implementation in (
            arguments.implementations
            or _DEFAULT_IMPLEMENTATIONS.get(family, _IMPLEMENTATIONS)
        )
    ]
    results: list[CaseResult] = []
    matrix_started = time.perf_counter()
    with MatrixConsole(output_directory / "matrix.log") as console:
        console.block(
            "MATRIX START",
            [
                f"UTC: {utc_now()}",
                f"OUTPUT: {output_directory}",
                f"REFERENCES: {reference_directory}",
                "CASES: "
                + ", ".join(
                    f"{implementation}_{family}"
                    for family, implementation in selected_cases
                ),
                f"COLD CACHES: {arguments.cold}",
                f"SEED: {arguments.seed}",
            ],
        )
        for ordinal, (family, implementation) in enumerate(selected_cases, start=1):
            progress = f"[{ordinal}/{len(selected_cases)}]"
            budget = overrides.get(family, DEFAULT_DEVICE_BUDGETS.get(family, 0))
            identity = f"{implementation}_{family}"
            case_log = output_directory / f"{identity}.log"
            case_log.unlink(missing_ok=True)
            reference = canonical_reference_path(
                reference_directory,
                model_name=family,
                implementation=implementation,
            )
            reference_state = (
                "regenerating"
                if arguments.regenerate_reference
                or not reference_artifact_exists(reference)
                else "reusing canonical"
            )
            started_at = utc_now()
            console.emit()
            console.block(
                f"CASE START {progress} {identity}",
                [
                    f"MODEL: {implementation}/{family}",
                    f"DEVICE BUDGET: {format_bytes(budget)}",
                    f"REFERENCE: {reference} ({reference_state})",
                    f"LOG: {case_log}",
                    f"START: {started_at}",
                ],
            )
            result = _run_case(
                family=family,
                implementation=implementation,
                device_budget=budget,
                output_directory=output_directory,
                reference_directory=reference_directory,
                environment=environment,
                regenerate_reference=arguments.regenerate_reference,
                seed=arguments.seed,
                model_config=arguments.model_config,
                data_geometry=arguments.data_geometry,
                case_factory=arguments.case_factory,
                case_options=arguments.case_option,
                optimizer_ordering=arguments.optimizer_ordering,
                cold=arguments.cold,
                cache_directory=arguments.cache_dir,
                detailed_artifacts=arguments.detailed_artifacts,
                console=console,
                progress=progress,
                case_log=case_log,
            )
            results.append(result)
            console.block(
                f"CASE {'PASS' if result.passed else 'FAIL'} {progress} {identity}",
                [
                    f"ARTIFACT: {result.artifact}",
                    f"START: {started_at}",
                    f"STOP: {utc_now()}",
                    f"DURATION: {result.elapsed_seconds:.3f} seconds",
                ],
            )
            if not result.passed and not arguments.keep_going:
                break

        summary = {
            "schema": "shadowspill.model_correctness_matrix/v1",
            "passed": len(results) == len(selected_cases)
            and all(item.passed for item in results),
            "cold": arguments.cold,
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
                }
                for item in results
            ],
        }
        summary_path = output_directory / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        console.emit()
        console.block(
            "MATRIX " + ("PASS" if summary["passed"] else "FAIL"),
            [
                "CASES PASSED: "
                f"{sum(1 for item in results if item.passed)}"
                f"/{len(selected_cases)}",
                f"SUMMARY: {summary_path}",
                f"STOP: {utc_now()}",
                f"DURATION: {time.perf_counter() - matrix_started:.3f} seconds",
            ],
        )
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
