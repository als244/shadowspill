"""Run one full-model ShadowSpill-only throughput qualification cell.

Separate workers build, profile, plan and execute the cell. :mod:`.verdict`
judges the execution results.
"""

from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import replace
from pathlib import Path

from benchmarking.spill import parse_remote, record
from qualification.device_defaults import (
    add_memory_budget_arguments,
    performance_defaults,
)
from qualification.pipeline.process import profile_and_plan, run_phase
from qualification.precision import add_dtype_arguments
from shadowspill.schema import artifact_schema

from .manifest import _planning_spill_budget
from .prepare import manifest_for_run


def _run(arguments: argparse.Namespace) -> dict[str, object]:
    manifest = manifest_for_run(arguments)
    budget = _planning_spill_budget(
        manifest, planning_spill_budget_gib=arguments.planning_spill_budget_gib
    )
    spill = None
    if arguments.remote_spill is not None:
        spill = parse_remote(arguments.remote_spill)
        budget = min(budget, spill.capacity)
        spill = replace(spill, capacity=budget)
    print("DTYPES: " + manifest.dtypes.description(), flush=True)
    output = arguments.output.expanduser().resolve()
    directory = output.with_suffix(".phases")
    directory.mkdir(parents=True, exist_ok=True)
    arguments.output = output
    (directory / "arguments.json").write_text(
        json.dumps(vars(arguments), default=str, indent=2)
    )
    (directory / "manifest.json").write_text(json.dumps(manifest.as_dict(), indent=2))
    (directory / "settings.json").write_text(
        json.dumps(
            {
                "execution_bytes": manifest.device_physical_capacity_bytes,
                "external_headroom_bytes": manifest.external_headroom_bytes,
                "spill_bytes": budget,
                "spill": record(spill),
            },
            indent=2,
        )
    )
    run_phase(directory, "build", "qualification.performance.prepare", directory)
    profile_and_plan(directory)
    if arguments.plan_only:
        from shadowspill import Plan

        from .verdict import plan_result

        return {
            **plan_result(Plan.load(directory / "plan.ssp"), directory),
            "schema": artifact_schema("full_model_qualification"),
            "manifest": manifest.as_dict(),
            "plan_only": True,
            "passed": True,
        }
    run_phase(directory, "execute", "qualification.performance.execute", directory)
    return json.loads(output.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=("llama3", "qwen35", "olmoe"))
    parser.add_argument("implementation", choices=("pytorch", "mlops"))
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--execution-budget-gib",
        type=int,
        help="total GPU allowance in GiB (default: 16; 10 below SM80)",
    )
    add_dtype_arguments(parser)
    add_memory_budget_arguments(parser)
    parser.add_argument("--seed", type=int, default=20_260_811)
    parser.add_argument("--groups", type=int, default=3)
    parser.add_argument("--steps-per-group", type=int, default=4)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--measure-only",
        action="store_true",
        help=(
            "report measurements without failing diagnostic gates; "
            "the artifact still records every gate field"
        ),
    )
    parser.add_argument(
        "--skip-checkpoint",
        action="store_true",
        help=(
            "run the throughput protocol without the streamed checkpoint/restore; "
            "this is a runtime probe, not checkpoint qualification"
        ),
    )
    parser.add_argument(
        "--profiler-annotations",
        action="store_true",
        help=(
            "emit profiler ranges around task boundaries and compiled calls, so an "
            "external profiler can attribute time to the task that spent it"
        ),
    )
    parser.add_argument(
        "--spill-budget-gib",
        type=int,
        help="override the manifest's runtime spill-pool capacity",
    )
    parser.add_argument(
        "--planning-spill-budget-gib",
        type=int,
        help="use a smaller spill capacity for both the saved Plan and execution",
    )
    parser.add_argument(
        "--remote-spill",
        metavar="HOST:PORT:BYTES",
        help=(
            "spill to a memory daemon on another machine instead of to pinned "
            "host memory. Everything else about the cell is unchanged, which "
            "is what makes the comparison mean something"
        ),
    )
    arguments = parser.parse_args()
    performance_defaults(arguments)
    if (
        arguments.execution_budget_gib is not None
        and arguments.execution_budget_gib <= 0
    ):
        parser.error("--execution-budget-gib must be positive")
    if arguments.groups <= 0 or arguments.steps_per_group <= 0:
        parser.error("groups and steps-per-group must be positive")
    if arguments.plan_only and arguments.skip_checkpoint:
        parser.error("--skip-checkpoint has no effect with --plan-only")
    if arguments.plan_only and arguments.measure_only:
        parser.error("--measure-only has nothing to measure with --plan-only")
    if arguments.spill_budget_gib is not None and arguments.spill_budget_gib <= 0:
        parser.error("--spill-budget-gib must be positive")
    if (
        arguments.planning_spill_budget_gib is not None
        and arguments.planning_spill_budget_gib <= 0
    ):
        parser.error("--planning-spill-budget-gib must be positive")
    try:
        result = _run(arguments)
    except BaseException as error:
        notes = tuple(str(note) for note in getattr(error, "__notes__", ()))
        failure = {
            "schema": artifact_schema("full_model_qualification_failure"),
            "family": arguments.family,
            "implementation": arguments.implementation,
            "error_type": type(error).__name__,
            "error": str(error),
            "error_notes": notes,
            "traceback": "".join(traceback.format_exception(error)),
        }
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.with_suffix(".failure.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n"
        )
        raise
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    identity = f"{arguments.implementation}_{arguments.family}"
    if arguments.measure_only:
        print(f"RESULT MEASURED: {identity}", flush=True)
    else:
        print(
            f"RESULT {'PASS' if result['passed'] else 'FAIL'}: {identity}", flush=True
        )
    if not arguments.plan_only:
        gates = (
            ("protocol_complete", "PROTOCOL"),
            ("objectives_finite", "OBJECTIVES"),
            ("logical_steps_passed", "LOGICAL STEPS"),
            ("physical_budget_passed", "PHYSICAL BUDGETS"),
            ("strict_runtime_passed", "STRICT RUNTIME"),
            ("simulator_gate_passed", "SIMULATOR"),
            ("regression_gate_passed", "REGRESSION"),
        )
        if not arguments.measure_only:
            for key, label in gates:
                if (
                    key == "regression_gate_passed"
                    and not result["regression_gate_applicable"]
                ):
                    print(
                        "  GATE REGRESSION: not applicable ("
                        + result["regression_comparison"]["reason"]
                        + ")"
                    )
                else:
                    print(f"  GATE {label}: {'pass' if result[key] else 'FAIL'}")
        print(
            f"  MEDIAN STEP: {result['median_step_seconds']:.4f} seconds "
            f"({result['median_tokens_per_second']:.1f} tokens/s)"
        )
        print(
            f"  PREDICTED STEP: {result['predicted_makespan_seconds']:.4f} "
            f"seconds (simulator error {result['simulator_relative_error']:+.2%})"
        )
        # Both ratios divide by throughput measured on the machine that set
        # the floors, so on any other machine they describe the hardware
        # rather than this run.  Measure-only reports the measurement itself.
        if not arguments.measure_only:
            ratio = result["regression_throughput_ratio"]
            if isinstance(ratio, float):
                print(f"  REGRESSION RATIO: {ratio:.2%}")
            # Reported, never gated: how close this run is to the predecessor
            # system ShadowSpill replaces. See workloads.full_model for the
            # provenance.
            ratio = result["predecessor_throughput_ratio"]
            if isinstance(ratio, float):
                print(f"  PREDECESSOR RATIO: {ratio:.2%}")
    print(f"  PLANNING: {result['planning_seconds']:.3f} seconds")
    print(f"  ARTIFACT: {arguments.output}", flush=True)
    if not result["passed"] and not arguments.plan_only and not arguments.measure_only:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
