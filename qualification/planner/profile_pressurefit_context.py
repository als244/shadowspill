"""Profile one saved PressureFit recomputation context without PyTorch."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from shadowspill.ir import Program, RecomputationSelection, ResidencySpec
from shadowspill.planner._native_portfolio import (
    NativeContextResult,
    decode_candidate_diagnostic,
    decode_schedule,
    evaluate_program_context_compiled,
)
from shadowspill.planner.model import (
    InitialPlacement,
    PressureFitDiagnostics,
    PressureFitOptions,
)
from shadowspill.simulator import SimulationConfig
from shadowspill.simulator._compiled import (
    CompiledSimulationTemplate,
    compile_simulation_template,
)
from shadowspill.simulator.model import DeviceSimulationConfig


def _load_program(path: Path) -> Program:
    return Program.from_dict(json.loads(path.read_text()))


def _load_selection(
    path: Path,
) -> tuple[
    tuple[ResidencySpec, ...],
    tuple[ResidencySpec, ...],
    tuple[RecomputationSelection, ...],
    SimulationConfig,
    PressureFitOptions,
    dict[str, Any],
]:
    value: dict[str, Any] = json.loads(path.read_text())
    initial = tuple(
        ResidencySpec.from_value(item, f"initial_residency[{index}]")
        for index, item in enumerate(value["initial_residency"])
    )
    final = tuple(
        ResidencySpec.from_value(item, f"final_residency[{index}]")
        for index, item in enumerate(value["final_residency"])
    )
    selections = tuple(
        RecomputationSelection.from_value(item, f"selections[{index}]")
        for index, item in enumerate(value["selections"])
    )
    simulation = value["simulation"]
    config = SimulationConfig(
        tuple(DeviceSimulationConfig(**item) for item in simulation["devices"]),
        simulation["host_capacity_bytes"],
    )
    options = value["options"]
    return (
        initial,
        final,
        selections,
        config,
        PressureFitOptions(
            initial_placement=InitialPlacement(options["initial_placement"]),
            residency_strategies=tuple(options["residency_strategies"]),
            prefetch_rules=tuple(options["prefetch_rules"]),
            evaluate_coalesced=options["evaluate_coalesced"],
            max_repair_attempts=options["max_repair_attempts"],
            workers=1,
        ),
        value,
    )


def _verify_frozen_selection(
    result: NativeContextResult,
    *,
    selection: dict[str, Any],
    template: CompiledSimulationTemplate,
) -> dict[str, object]:
    diagnostics = PressureFitDiagnostics.from_value(
        selection["diagnostics"], "selection.diagnostics"
    )
    selection_id = diagnostics.selected_selection_id
    expected_context = next(
        item
        for item in diagnostics.recomputation_contexts
        if item.selection_id == selection_id
    )
    expected_candidates = expected_context.candidate_evaluations
    actual_candidates = tuple(
        decode_candidate_diagnostic(
            item,
            selection_id=selection_id,
            simulation=template,
        )
        for item in result.candidates
    )
    if len(expected_candidates) != len(actual_candidates):
        raise AssertionError(
            "saved selection contains "
            f"{len(expected_candidates)} candidates for its winning context; "
            f"the planner returned {len(actual_candidates)}"
        )
    candidate_mismatches = tuple(
        index
        for index, (actual, expected) in enumerate(
            zip(actual_candidates, expected_candidates, strict=True)
        )
        if _without_work_times(actual.to_dict())
        != _without_work_times(expected.to_dict())
    )
    schedule_equal = (
        result.selected_schedule is not None
        and decode_schedule(result.selected_schedule, template).to_dict()
        == selection["schedule"]
    )
    if candidate_mismatches or not schedule_equal:
        raise AssertionError(
            "compiled planner diverged from the frozen selection: "
            f"candidate_mismatches={candidate_mismatches}, "
            f"schedule_equal={schedule_equal}"
        )
    return {
        "candidate_diagnostics_equal": True,
        "selected_schedule_equal": True,
    }


def _summary(result: NativeContextResult, wall_seconds: float) -> dict[str, object]:
    return {
        "wall_seconds": wall_seconds,
        "candidate_count": len(result.candidates),
        "valid_candidate_count": sum(item.status == 0 for item in result.candidates),
        "repairs": result.repairs.to_dict(),
        "work": result.work.to_dict(),
        "selected_candidate_index": result.selected_candidate_index,
        "selected_makespan_ns": result.selected_makespan_ns,
    }


def _without_work_times(value: object) -> object:
    if isinstance(value, list):
        return [_without_work_times(item) for item in value]
    if not isinstance(value, dict):
        return value
    return {
        key: _without_work_times(item)
        for key, item in value.items()
        if not key.endswith("_time_ns")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program", type=Path)
    parser.add_argument("selection", type=Path)
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="accept a compact context file without frozen diagnostics",
    )
    arguments = parser.parse_args()

    program = _load_program(arguments.program)
    initial, final, selections, config, options, selection = _load_selection(
        arguments.selection
    )
    selected_tasks = program.selected_tasks(selections)
    template = compile_simulation_template(
        program,
        selections,
        config,
        selected_tasks=selected_tasks,
        initial_residency=initial,
        final_residency=final,
    )
    started = time.perf_counter()
    result = evaluate_program_context_compiled(template, options)
    wall_seconds = time.perf_counter() - started
    if result is None:
        raise RuntimeError("the compiled planner rejected the saved context")
    summary = _summary(result, wall_seconds)
    if not arguments.skip_verification:
        summary.update(
            _verify_frozen_selection(
                result,
                selection=selection,
                template=template,
            )
        )
    print(json.dumps(summary, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
