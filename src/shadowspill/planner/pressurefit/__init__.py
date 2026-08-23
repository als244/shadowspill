"""PressureFit: the best schedule for one Program under one budget.

The work divides four ways, one module each:

``recomputation``  which recomputation selections are worth evaluating
``search``         evaluating them and picking a winner
``refinement``     what to try when admission refuses that winner
this module        the entry point, and validating what it was given
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace

from shadowspill.ir import (
    Program,
    ResidencySpec,
    shared_residency_footprint,
)
from shadowspill.simulator import SimulationConfig
from shadowspill.simulator.capi import simulator_api

from ..admission import AdmissionFacts
from ..capi import planner_api
from ..diagnostics import AdmissionRefinement
from ..recomputation import build_recomputation_portfolio
from ..request import PressureFitOptions
from ..result import PressureFitInfeasibleError, PressureFitResult
from .refinement import (
    round_up_admission_reserve,
    scheduled_admission_refinement,
    with_object_capacity,
)
from .search import (
    build_problems,
    finish_pressurefit,
    preflight_problems,
    run_problems,
    worker_count,
)


def validate_pressurefit_inputs(
    program: Program,
    initial_residency: tuple[ResidencySpec, ...],
    final_residency: tuple[ResidencySpec, ...],
    config: SimulationConfig,
    admission: AdmissionFacts | None,
) -> None:
    """Validate public types and the logical/physical capacity relationship."""

    if not isinstance(program, Program):
        raise TypeError("program must be a Program")
    if not isinstance(initial_residency, tuple):
        raise TypeError("initial_residency must be a tuple")
    if not isinstance(final_residency, tuple):
        raise TypeError("final_residency must be a tuple")
    if not isinstance(config, SimulationConfig):
        raise TypeError("config must be a SimulationConfig")
    if admission is None:
        return
    if not isinstance(admission, AdmissionFacts):
        raise TypeError("admission must be an AdmissionFacts")
    admission.validate(program)
    configured = {item.device_id: item for item in config.devices}
    shared = shared_residency_footprint(program)
    movable_capacity = (
        configured[admission.device_id].capacity_bytes
        - shared.for_device(admission.device_id)
        if admission.device_id in configured
        else None
    )
    if (
        admission.device_id not in configured
        or movable_capacity != admission.object_capacity_bytes
    ):
        raise ValueError(
            "feasibility capacity after shared residency must equal "
            "AdmissionFacts object capacity"
        )


def validate_schedule_feasibility(
    program: Program,
    *,
    initial_residency: tuple[ResidencySpec, ...],
    final_residency: tuple[ResidencySpec, ...] = (),
    config: SimulationConfig,
    admission: AdmissionFacts | None = None,
) -> None:
    """Reject irreducible capacity failures using the planner."""

    validate_pressurefit_inputs(
        program,
        initial_residency,
        final_residency,
        config,
        admission,
    )
    simulator_api()
    planner_api()
    problems = build_problems(
        program,
        initial_residency,
        final_residency,
        config,
        admission,
        portfolio=build_recomputation_portfolio(program),
        progress=None,
    )
    preflight_problems(problems)


def pressurefit_once(
    program: Program,
    *,
    initial_residency: tuple[ResidencySpec, ...],
    final_residency: tuple[ResidencySpec, ...] = (),
    config: SimulationConfig,
    options: PressureFitOptions | None = None,
    admission: AdmissionFacts | None = None,
    progress: Callable[[str], None] | None = None,
) -> PressureFitResult:
    """Evaluate every selection through the planner."""

    validate_pressurefit_inputs(
        program,
        initial_residency,
        final_residency,
        config,
        admission,
    )
    simulator_api()
    planner_api()

    selected_options = options or PressureFitOptions()
    portfolio = build_recomputation_portfolio(program)
    if progress is not None:
        progress(
            "PressureFit portfolio: "
            f"groups={len(program.recomputation_groups)}, "
            f"selections={len(portfolio)}"
        )
    started = time.perf_counter_ns()
    problems = preflight_problems(
        build_problems(
            program,
            initial_residency,
            final_residency,
            config,
            admission,
            portfolio=portfolio,
            progress=progress,
        )
    )
    results = run_problems(problems, selected_options)
    valid_pairs = tuple(
        (problem, result)
        for problem, result in zip(problems, results, strict=True)
        if result is not None
    )
    if progress is not None:
        progress(
            "PressureFit compiled problems and candidates finished: "
            f"valid={len(valid_pairs)}/{len(problems)}, "
            "candidates="
            f"{sum(len(result.candidates) for _problem, result in valid_pairs)}, "
            f"workers={worker_count(selected_options, len(problems))}, "
            f"elapsed={(time.perf_counter_ns() - started) / 1e9:.3f}s"
        )
    if not valid_pairs:
        raise RuntimeError(
            "PressureFit rejected every selection after semantic "
            "feasibility validation succeeded"
        )
    return finish_pressurefit(
        program,
        initial_residency,
        final_residency,
        config,
        selected_options,
        tuple(problem for problem, _result in valid_pairs),
        tuple(result for _problem, result in valid_pairs),
        admission,
    )


def pressurefit(
    program: Program,
    *,
    initial_residency: tuple[ResidencySpec, ...],
    final_residency: tuple[ResidencySpec, ...] = (),
    config: SimulationConfig,
    options: PressureFitOptions | None = None,
    admission: AdmissionFacts | None = None,
    progress: Callable[[str], None] | None = None,
) -> PressureFitResult:
    """Select a schedule and refine dynamic-slab headroom."""

    original_config = config
    current_config = config
    current_admission = admission
    refinements: list[AdmissionRefinement] = []
    while True:
        try:
            result = pressurefit_once(
                program,
                initial_residency=initial_residency,
                final_residency=final_residency,
                config=current_config,
                options=options,
                admission=current_admission,
                progress=progress,
            )
            effective_capacity = (
                None
                if current_admission is None
                else current_admission.object_capacity_bytes
            )
            return replace(
                result,
                simulation_config=original_config,
                diagnostics=replace(
                    result.diagnostics,
                    admission_refinements=tuple(refinements),
                    effective_object_capacity_bytes=effective_capacity,
                ),
                admission_facts=admission,
            )
        except PressureFitInfeasibleError as error:
            if (
                current_admission is None
                or error.kind != "physical_admission"
                or error.required_bytes is None
                or error.required_bytes <= 0
            ):
                raise
            previous = current_admission.object_capacity_bytes
            scheduled_increment = scheduled_admission_refinement(len(refinements))
            increment = max(
                round_up_admission_reserve(error.required_bytes),
                scheduled_increment,
            )
            capacity = previous - increment
            if capacity <= 0:
                raise
            refinements.append(
                AdmissionRefinement(
                    attempt=len(refinements) + 1,
                    previous_object_capacity_bytes=previous,
                    required_additional_slack_bytes=error.required_bytes,
                    reserve_increment_bytes=increment,
                    object_capacity_bytes=capacity,
                )
            )
            if progress is not None:
                progress(
                    "PressureFit physical admission refinement "
                    f"{len(refinements)}: required_slack={error.required_bytes}, "
                    f"reserve_increment={increment}, object_capacity={capacity}"
                )
            current_config, current_admission = with_object_capacity(
                current_config,
                current_admission,
                capacity,
                shared_execution_bytes=shared_residency_footprint(program).for_device(
                    current_admission.device_id
                ),
            )


__all__ = ["pressurefit", "validate_schedule_feasibility"]
