"""Launch the model-scale numerical and physical-budget qualification matrix."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from qualification.numerical.cases import DEFAULT_DEVICE_BUDGETS

_FAMILIES: Final = tuple(DEFAULT_DEVICE_BUDGETS)
_IMPLEMENTATIONS: Final = ("pytorch", "mlops")
_LIBRARIES: Final = {
    "SHADOWSPILL_PYTORCH_LIBRARY": "libshadowspill_pytorch.so",
    "SHADOWSPILL_SIMULATOR_LIBRARY": "libshadowspill_simulator.so",
    "SHADOWSPILL_PLANNER_LIBRARY": "libshadowspill_planner.so",
}


@dataclass(frozen=True, slots=True)
class CaseResult:
    family: str
    implementation: str
    device_budget_bytes: int
    elapsed_seconds: float
    return_code: int
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
        if separator == "" or not family or (
            valid_models is not None and family not in valid_models
        ):
            raise argparse.ArgumentTypeError(
                "budget must be MODEL=BYTES for one of "
                + ", ".join(sorted(valid_models or ()))
            )
        result[family] = _parse_bytes(budget)
    return result


def _environment(
    *, build_directory: Path | None, cache_directory: Path | None
) -> dict[str, str]:
    environment = dict(os.environ)
    if build_directory is not None:
        resolved = build_directory.expanduser().resolve()
        for variable, filename in _LIBRARIES.items():
            library = resolved / filename
            if not library.is_file():
                raise FileNotFoundError(f"missing {variable} library: {library}")
            environment[variable] = str(library)
    missing = [name for name in _LIBRARIES if name not in environment]
    if missing:
        raise RuntimeError(
            "compiled libraries are not installed or configured; pass --build-dir "
            "or set " + ", ".join(missing)
        )
    if cache_directory is not None:
        resolved_cache = cache_directory.expanduser().resolve()
        resolved_cache.mkdir(parents=True, exist_ok=True)
        environment["SHADOWSPILL_PROFILE_CACHE"] = str(resolved_cache / "profiles")
        environment["SHADOWSPILL_RECOMPUTATION_CACHE"] = str(
            resolved_cache / "recomputation"
        )
    return environment


def _run_case(
    *,
    family: str,
    implementation: str,
    device_budget: int,
    output_directory: Path,
    environment: dict[str, str],
    reuse_eager: bool,
    seed: int,
    model_config: str,
    data_geometry: str | None,
    case_factory: str | None,
    case_options: list[str],
) -> CaseResult:
    prefix = f"{implementation}_{family}"
    reference = output_directory / f"{prefix}_eager.pt"
    artifact = output_directory / f"{prefix}.json"
    base = [sys.executable, "-m", "qualification.numerical.run"]
    options = ["--seed", str(seed), "--model-config", model_config]
    if data_geometry is not None:
        options.extend(("--data-geometry", data_geometry))
    if case_factory is not None:
        options.extend(("--case-factory", case_factory))
    for value in case_options:
        options.extend(("--case-option", value))
    commands: list[list[str]] = []
    if not reuse_eager or not reference.is_file():
        commands.append(
            [
                *base,
                "_eager",
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
    for command in commands:
        completed = subprocess.run(command, check=False, env=environment)
        return_code = completed.returncode
        if return_code != 0:
            break
    passed = False
    if return_code == 0 and artifact.is_file():
        payload = json.loads(artifact.read_text())
        passed = bool(
            payload.get("schema") == "shadowspill.numerical_qualification/v3"
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
        artifact=str(artifact),
        passed=passed,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run fresh-process eager/planned parity, checkpoint replay, transfer, "
            "recomputation, and physical-budget gates."
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
        default=_IMPLEMENTATIONS,
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
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--seed", type=int, default=20_260_811)
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
        help="verification-case factory for custom model names",
    )
    parser.add_argument(
        "--case-option",
        action="append",
        default=[],
        metavar="NAME=JSON",
        help="repeatable custom-factory argument",
    )
    parser.add_argument(
        "--reuse-eager",
        action="store_true",
        help="reuse an existing eager artifact; regeneration is safer by default",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="continue remaining cases after a failed case",
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
        missing_budgets = [
            name
            for name in custom_names
            if name not in overrides
        ]
        if missing_budgets:
            raise RuntimeError(
                "custom model budgets must be explicit with --budget: "
                + ", ".join(missing_budgets)
            )
        environment = _environment(
            build_directory=arguments.build_dir,
            cache_directory=arguments.cache_dir,
        )
    except (argparse.ArgumentTypeError, FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))

    output_directory = arguments.output_dir.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    results: list[CaseResult] = []
    for family in arguments.models:
        for implementation in arguments.implementations:
            budget = overrides.get(family, DEFAULT_DEVICE_BUDGETS.get(family, 0))
            print(
                f"\n== {family}/{implementation}: {budget} physical bytes ==",
                flush=True,
            )
            result = _run_case(
                family=family,
                implementation=implementation,
                device_budget=budget,
                output_directory=output_directory,
                environment=environment,
                reuse_eager=arguments.reuse_eager,
                seed=arguments.seed,
                model_config=arguments.model_config,
                data_geometry=arguments.data_geometry,
                case_factory=arguments.case_factory,
                case_options=arguments.case_option,
            )
            results.append(result)
            print(
                f"{family}/{implementation}: "
                f"{'PASS' if result.passed else 'FAIL'} "
                f"({result.elapsed_seconds:.3f}s)",
                flush=True,
            )
            if not result.passed and not arguments.keep_going:
                break
        if results and not results[-1].passed and not arguments.keep_going:
            break

    summary = {
        "schema": "shadowspill.model_correctness_matrix/v1",
        "passed": len(results)
        == len(arguments.models) * len(arguments.implementations)
        and all(item.passed for item in results),
        "cases": [
            {
                "family": item.family,
                "implementation": item.implementation,
                "device_budget_bytes": item.device_budget_bytes,
                "elapsed_seconds": item.elapsed_seconds,
                "return_code": item.return_code,
                "artifact": item.artifact,
                "passed": item.passed,
            }
            for item in results
        ],
    }
    summary_path = output_directory / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"\nSummary: {summary_path}", flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
