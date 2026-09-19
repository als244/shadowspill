"""Run one full-model ShadowSpill-only throughput qualification cell.

:mod:`.phases` runs the cell, :mod:`.verdict` judges what it produced, and
``main`` below is the command line both are reached through.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from shadowspill.schema import artifact_schema
from tools.qualification.model_state import release_case_model
from tools.qualification.runtime_evidence import (
    check_physical_budget,
)
from workloads.full_model import build_case

from .manifest import (
    _manifest_with_overrides,
    _planning_spill_budget,
)
from .phases import (
    _announce_prediction,
    _calibrated_runtime,
    _measure_groups,
    _plan_case,
    _warm_step,
)
from .verdict import _measured_result, _plan_only_result


def _remote_spill_pool(arguments: argparse.Namespace) -> object | None:
    """The peer's pool named by ``--remote-spill``, or ``None`` for pinned host.

    Parsed here rather than in the matrix so a cell run by hand behaves exactly
    as one the matrix spawned, which is the whole reason the option travels as
    a string. Imported lazily: a local run should not load the network library
    to decide it does not need it.
    """

    value = getattr(arguments, "remote_spill", None)
    if value is None:
        return None
    host, _, rest = value.partition(":")
    port, _, size = rest.partition(":")
    if not host or not port.isdigit() or not size.isdigit():
        raise SystemExit(f"--remote-spill must read HOST:PORT:BYTES, not {value!r}")
    from shadowspill.network import remote

    return remote(capacity=int(size), host=host, port=int(port))


def _run(arguments: argparse.Namespace) -> dict[str, object]:
    manifest = _manifest_with_overrides(
        arguments.family,
        arguments.implementation,
        spill_budget_gib=arguments.spill_budget_gib,
    )
    planning_spill_budget = _planning_spill_budget(
        manifest,
        planning_spill_budget_gib=arguments.planning_spill_budget_gib,
    )
    output = arguments.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    cache = (
        arguments.artifact_store.expanduser().resolve()
        if arguments.artifact_store is not None
        else output.parent / "artifact_store" / manifest.identity
    )
    runtime, capabilities, calibration_attempts = _calibrated_runtime(
        manifest, _remote_spill_pool(arguments)
    )
    case = build_case(manifest, seed=arguments.seed, runtime=runtime)
    with case.implementations():
        planned = _plan_case(
            case,
            manifest,
            arguments,
            runtime=runtime,
            cache=cache,
            output=output,
            planning_spill_budget=planning_spill_budget,
        )
        physical_statuses = [check_physical_budget()]
        if arguments.plan_only:
            result = _plan_only_result(
                manifest,
                planned,
                physical_statuses,
                planning_spill_budget=planning_spill_budget,
                capabilities=capabilities,
            )
        else:
            warm = _warm_step(planned.training, case, arguments, physical_statuses)
            _announce_prediction(manifest, planned.report, runtime)
            measured = _measure_groups(
                planned.training, case, manifest, arguments, physical_statuses
            )
            result = _measured_result(
                manifest,
                arguments,
                planned,
                warm,
                measured,
                runtime=runtime,
                physical_statuses=physical_statuses,
                planning_spill_budget=planning_spill_budget,
                capabilities=capabilities,
                calibration_attempts=calibration_attempts,
            )
        planned.training.close()
        # Qualification never reuses the model; an export copy would stack an
        # anonymous full-model allocation on the registered spill arena.
        release_case_model(case, runtime=runtime)
        runtime.close()
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=("llama3", "qwen35", "olmoe"))
    parser.add_argument("implementation", choices=("pytorch", "mlops"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=20_260_811)
    parser.add_argument("--groups", type=int, default=3)
    parser.add_argument("--steps-per-group", type=int, default=4)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--measure-only",
        action="store_true",
        help=(
            "report throughput without judging it; the gates compare against "
            "floors measured on one machine, so they carry no meaning on "
            "another. The artifact still records every gate field"
        ),
    )
    parser.add_argument(
        "--skip-checkpoint",
        action="store_true",
        help=(
            "run the throughput protocol without the anonymous full-state "
            "checkpoint copy; this is a runtime probe, not checkpoint qualification"
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
    parser.add_argument("--artifact-store", type=Path)
    parser.add_argument("--build-store", type=Path)
    parser.add_argument("--plan-store", type=Path)
    for tree in ("build", "plan"):
        parser.add_argument(
            f"--{tree}-store-mode",
            choices=("contribute", "reuse", "require", "refresh"),
            default="contribute",
        )
    parser.add_argument("--export-bypass-key")
    parser.add_argument(
        "--spill-budget-gib",
        type=int,
        help="override the manifest's runtime spill-pool capacity",
    )
    parser.add_argument(
        "--planning-spill-budget-gib",
        type=int,
        help="use a smaller planning budget within the runtime spill pool",
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
