"""Run the selected numerical cases and write one matrix summary."""

from __future__ import annotations

import json
import os
import time

from qualification.device_defaults import numerical_defaults
from qualification.matrix_logging import MatrixConsole, utc_now
from qualification.numerical.spill import configured_spill, spill_description
from qualification.precision import dtype_overrides
from shadowspill.memory import SpillPool

from .arguments import (
    DEFAULT_IMPLEMENTATIONS,
    IMPLEMENTATIONS,
    budget_overrides_for_args,
    create_parser,
)
from .case import run_case
from .commands import CaseOptions
from .report import CaseResult, announce_case, matrix_summary


def main(spill: SpillPool | None = None) -> int:
    """The matrix, spilling wherever `spill` says.

    Host, SSD and remote gates share cases, references, tolerances and verdicts.
    """

    parser = create_parser()
    arguments = parser.parse_args()
    selected_spill = configured_spill(parser, arguments)
    if spill is not None and selected_spill is not None:
        parser.error("this gate already supplies its spill pool")
    selected_spill = spill or selected_spill
    budgets = numerical_defaults(arguments)
    overrides = budget_overrides_for_args(parser, arguments)
    options = CaseOptions(
        environment=dict(os.environ),
        external_headroom_mib=arguments.external_headroom_mib,
        reject_overbudget=arguments.reject_overbudget,
        reference_directory=arguments.reference_dir.expanduser().resolve(),
        regenerate_reference=arguments.regenerate_reference,
        seed=arguments.seed,
        model_config=arguments.model_config,
        **dtype_overrides(arguments),
        data_geometry=arguments.data_geometry,
        case_factory=arguments.case_factory,
        case_options=arguments.case_option,
        optimizer_ordering=arguments.optimizer_ordering,
        data_ordering=arguments.data_ordering,
        empty_caches=arguments.empty_caches,
        cache_directory=arguments.cache_dir,
        detailed_artifacts=arguments.detailed_artifacts,
        retain_execution_files=arguments.retain_execution_files,
        spill_pool=selected_spill,
    )
    output_directory = arguments.output_dir.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    options.reference_directory.mkdir(parents=True, exist_ok=True)
    selected_cases = [
        (family, implementation)
        for family in arguments.models
        for implementation in (
            arguments.implementations
            or DEFAULT_IMPLEMENTATIONS.get(family, IMPLEMENTATIONS)
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
                "SPILL: "
                + json.dumps(spill_description(selected_spill), sort_keys=True),
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
            budget = overrides.get(family, budgets.get(family, 0))
            identity = f"{implementation}_{family}"
            case_log = output_directory / f"{identity}.log"
            case_log.unlink(missing_ok=True)
            started_at = announce_case(
                console,
                options,
                family=family,
                implementation=implementation,
                budget=budget,
                progress=progress,
                case_log=case_log,
            )
            result = run_case(
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

        summary = matrix_summary(
            results, selected_cases, empty_caches=options.empty_caches
        )
        summary["spill_pool"] = spill_description(selected_spill)
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
