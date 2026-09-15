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

from shadowspill.schema import artifact_schema
from workloads.numerical import DEFAULT_DEVICE_BUDGETS

from ..matrix_logging import MatrixConsole, format_bytes, utc_now
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
    failure_categories: tuple[str, ...] = ()
    failures: tuple[str, ...] = ()


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


@dataclass(frozen=True, slots=True)
class _CaseOptions:
    """What every case in the matrix is run the same way with."""

    environment: dict[str, str]
    reference_directory: Path
    regenerate_reference: bool
    seed: int
    model_config: str
    data_geometry: str | None
    case_factory: str | None
    case_options: list[str]
    optimizer_ordering: str
    data_ordering: str | None
    empty_caches: bool
    cache_directory: Path | None
    detailed_artifacts: bool

    def case_arguments(self) -> list[str]:
        """The options both arms of a case are given, in their fixed order."""

        arguments = [
            "--seed",
            str(self.seed),
            "--model-config",
            self.model_config,
            "--optimizer-ordering",
            self.optimizer_ordering,
        ]
        if self.data_geometry is not None:
            arguments.extend(("--data-geometry", self.data_geometry))
        if self.data_ordering is not None:
            arguments.extend(("--data-ordering", self.data_ordering))
        if self.case_factory is not None:
            arguments.extend(("--case-factory", self.case_factory))
        for value in self.case_options:
            arguments.extend(("--case-option", value))
        return arguments


def _case_commands(
    family: str,
    implementation: str,
    device_budget: int,
    reference: Path,
    artifact: Path,
    options: _CaseOptions,
) -> list[list[str]]:
    """The reference arm, when one must be made, and always the planned arm."""

    base = [sys.executable, "-m", "tools.qualification.numerical"]
    shared = options.case_arguments()
    commands: list[list[str]] = []
    if options.regenerate_reference or not reference_artifact_exists(reference):
        commands.append(
            [
                *base,
                "_reference",
                family,
                str(reference),
                "--model-implementation",
                implementation,
                *shared,
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
            *shared,
        ]
    )
    return commands


def _run_commands(
    commands: list[list[str]],
    *,
    prefix: str,
    options: _CaseOptions,
    output_directory: Path,
    console: MatrixConsole,
    progress: str,
    case_log: Path,
) -> int:
    """Run each arm in its own process, in the caches the options ask for."""

    return_code = 0
    for command_index, command in enumerate(commands):
        command_environment = dict(options.environment)
        is_reference = command_index == 0 and len(commands) == 2
        plan_cache = (
            (output_directory / "artifact_store" / prefix)
            if options.cache_directory is None
            else options.cache_directory.expanduser().resolve() / prefix
        )
        cache_root: Path | None = None
        if options.empty_caches:
            cache_parent = (
                output_directory / ".empty_caches"
                if options.cache_directory is None
                else options.cache_directory.expanduser().resolve()
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
            command.extend(("--artifact-store", str(plan_cache)))
            if options.detailed_artifacts:
                command.append("--detailed-artifacts")
            else:
                # Nothing to keep from a cell that only has to agree, so it
                # reads what is there and leaves the store as it found it.
                command.extend(("--plan-store-mode", "reuse"))
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
    return return_code


def _case_verdict(
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


def _run_case(
    *,
    family: str,
    implementation: str,
    device_budget: int,
    output_directory: Path,
    options: _CaseOptions,
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
        _case_commands(
            family, implementation, device_budget, reference, artifact, options
        ),
        prefix=prefix,
        options=options,
        output_directory=output_directory,
        console=console,
        progress=progress,
        case_log=case_log,
    )
    passed, return_code, failure_categories, failures = _case_verdict(
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


def _parser() -> argparse.ArgumentParser:
    """Which cells to run, what to run them with, and where to put them."""

    parser = argparse.ArgumentParser(
        description=(
            "Run fresh-process compiled-reference/planned parity, checkpoint "
            "replay, transfer, and physical-budget gates while reporting "
            "graph-pair selections diagnostically."
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
        "--empty-caches",
        action="store_true",
        help=(
            "start every subprocess with an empty artifact store and empty "
            "Inductor and Triton caches, in a temporary directory removed "
            "afterwards, so nothing is reused from an earlier run or another "
            "case"
        ),
    )
    parser.add_argument("--seed", type=int, default=20_260_811)
    parser.add_argument(
        "--optimizer-ordering",
        choices=("stage_interleaved", "tail"),
        default="stage_interleaved",
        help=(
            "group optimizer updates by stage and place them at their gradient frontier"
        ),
    )
    parser.add_argument(
        "--data-ordering",
        help="how every selected case walks its microbatches, as"
        " <depth>x<breadth> with r for the reversed backward walk and p for"
        " the paired loss, for example 2x4rp; the product must be the case's"
        " microbatch count. Omitted plans depth-first, which is what every"
        " stored reference was compared against so far",
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
            "evidence is the default"
        ),
    )
    return parser


def _budgets(
    parser: argparse.ArgumentParser, arguments: argparse.Namespace
) -> dict[str, int]:
    """Every selected model must have a name and a budget before anything runs."""

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
    except (argparse.ArgumentTypeError, FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))
    return overrides


def _summary(
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


def main() -> int:
    parser = _parser()
    arguments = parser.parse_args()
    overrides = _budgets(parser, arguments)
    options = _CaseOptions(
        environment=dict(os.environ),
        reference_directory=arguments.reference_dir.expanduser().resolve(),
        regenerate_reference=arguments.regenerate_reference,
        seed=arguments.seed,
        model_config=arguments.model_config,
        data_geometry=arguments.data_geometry,
        case_factory=arguments.case_factory,
        case_options=arguments.case_option,
        optimizer_ordering=arguments.optimizer_ordering,
        data_ordering=arguments.data_ordering,
        empty_caches=arguments.empty_caches,
        cache_directory=arguments.cache_dir,
        detailed_artifacts=arguments.detailed_artifacts,
    )
    output_directory = arguments.output_dir.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    options.reference_directory.mkdir(parents=True, exist_ok=True)
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
                f"REFERENCES: {options.reference_directory}",
                "CASES: "
                + ", ".join(
                    f"{implementation}_{family}"
                    for family, implementation in selected_cases
                ),
                f"EMPTY CACHES: {options.empty_caches}",
                f"SEED: {options.seed}",
            ],
        )
        for ordinal, (family, implementation) in enumerate(selected_cases, start=1):
            progress = f"[{ordinal}/{len(selected_cases)}]"
            budget = overrides.get(family, DEFAULT_DEVICE_BUDGETS.get(family, 0))
            identity = f"{implementation}_{family}"
            case_log = output_directory / f"{identity}.log"
            case_log.unlink(missing_ok=True)
            started_at = _announce_case(
                console,
                options,
                family=family,
                implementation=implementation,
                budget=budget,
                progress=progress,
                case_log=case_log,
            )
            result = _run_case(
                family=family,
                implementation=implementation,
                device_budget=budget,
                output_directory=output_directory,
                options=options,
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
                    *(
                        [f"FAILED: {', '.join(result.failure_categories)}"]
                        + [f"  {detail}" for detail in result.failures]
                        if result.failures
                        else []
                    ),
                ],
            )
            if not result.passed and not arguments.keep_going:
                break

        summary = _summary(results, selected_cases, empty_caches=options.empty_caches)
        summary_path = output_directory / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        console.emit()
        console.block(
            "MATRIX " + ("PASS" if summary["passed"] else "FAIL"),
            [
                "CASES PASSED: "
                f"{sum(1 for item in results if item.passed)}"
                f"/{len(selected_cases)}",
                *(
                    f"{item.family}/{item.implementation}: "
                    f"{', '.join(item.failure_categories)}"
                    for item in results
                    if not item.passed
                ),
                f"SUMMARY: {summary_path}",
                f"STOP: {utc_now()}",
                f"DURATION: {time.perf_counter() - matrix_started:.3f} seconds",
            ],
        )
    return 0 if summary["passed"] else 1


def _announce_case(
    console: MatrixConsole,
    options: _CaseOptions,
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
            f"REFERENCE: {reference} ({reference_state})",
            f"LOG: {case_log}",
            f"START: {started_at}",
        ],
    )
    return started_at


if __name__ == "__main__":
    raise SystemExit(main())
