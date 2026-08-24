"""Search across resolved programs, one at a time.

A Program arrives with alternatives still open: each graph pair can be saved
or recomputed, so one Program expands into several *resolved programs*, each
a concrete task set. PressureFit plans one of those. Deciding which ones to
try, and in what order, is this layer's job.

The split matters because of what will cross the boundary. A plan admitted
under any resolved program is a real plan, so it can bound the search under
every other one -- which is what makes the order worth choosing. PressureFit
receives that bound as an argument and never learns where it came from, so
it keeps knowing only tasks, runtimes, object accesses, budgets and
bandwidths.

This module is framework-neutral: it needs a Program, its resolutions and a
budget, and nothing that belongs to a frontend.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from shadowspill.ir import Program, ResidencySpec, shared_residency_footprint
from shadowspill.simulator import SimulationConfig
from shadowspill.simulator.capi import simulator_api

from .admission import AdmissionFacts
from .best import BestFound
from .capi import planner_api
from .diagnostics import AdmissionRefinement
from .pressurefit import evaluate_resolution, validate_pressurefit_inputs
from .pressurefit.candidates import CProblemResult
from .pressurefit.refinement import (
    round_up_admission_reserve,
    scheduled_admission_refinement,
    with_object_capacity,
)
from .pressurefit.search import (
    SelectionProblem,
    build_problems,
    finish_pressurefit,
    preflight_problems,
    worker_count,
)
from .recomputation import Resolution, resolutions
from .request import PressureFitOptions
from .result import PressureFitInfeasibleError, PressureFitResult

#: What a graph pair's alternatives are called today. Ordering only needs to
#: recognise the two extremes; anything else falls through to the middle.
_RECOMPUTE = "recompute"
_SAVE = "save"


def _recompute_share(resolution: Resolution) -> float:
    """Fraction of this resolution's graph pairs that recompute."""

    if not resolution:
        return 0.0
    recomputed = sum(1 for item in resolution if item.option_id == _RECOMPUTE)
    return recomputed / len(resolution)


def ordered_resolutions(program: Program) -> tuple[Resolution, ...]:
    """Return the resolved programs to try, most-likely-to-admit first.

    Order is part of the algorithm, not a detail of it. A plan admitted under
    any resolved program bounds the search under every later one, so the
    order decides how much work the search does — and, more sharply, whether
    the bound exists early enough to prevent any work at all.

    1. **Everything recomputed.** Minimal simultaneous residency, so if any
       resolution admits, this one does. It buys an incumbent, however slow
       that incumbent is, and an incumbent is what makes everything after it
       cheap.
    2. **Nothing recomputed.** Minimal compute and no added work, so on an
       unpressured program it is immediately optimal. It either wins outright
       or is rejected quickly against the bound from step 1.
    3. **The rest**, now facing a bound from one or both extremes.

    "Fits if anything does" is the likely case rather than a guarantee: peak
    working space rises when recomputing, so a program could in principle
    fail on workspace at full recompute while fitting with some saved. The
    order is a heuristic about where to look first and never a claim about
    what is feasible, so being wrong costs a little work and nothing else.
    """

    resolved = resolutions(program)
    if len(resolved) < 2:
        return resolved
    shares = {id(item): _recompute_share(item) for item in resolved}
    most = max(resolved, key=lambda item: (shares[id(item)], resolved.index(item)))
    least = min(resolved, key=lambda item: (shares[id(item)], resolved.index(item)))
    if most is least:
        return resolved
    middle = tuple(item for item in resolved if item is not most and item is not least)
    return (most, least, *middle)


def plan_program(
    program: Program,
    *,
    initial_residency: tuple[ResidencySpec, ...],
    final_residency: tuple[ResidencySpec, ...] = (),
    config: SimulationConfig,
    options: PressureFitOptions | None = None,
    admission: AdmissionFacts | None = None,
    best: BestFound | None = None,
    progress: Callable[[str], None] | None = None,
) -> PressureFitResult:
    """Plan `program` by planning each of its resolved programs in turn.

    `best` carries a plan already in hand across resolved programs, so each
    one is searched against the answer the previous ones found. Passing one
    in shares that bound with a wider search; omitting it means this call
    starts from nothing and keeps its own.
    """

    validate_pressurefit_inputs(
        program,
        initial_residency,
        final_residency,
        config,
        admission,
    )
    selected_options = options or PressureFitOptions()
    resolved = ordered_resolutions(program)
    if progress is not None:
        progress(
            "PressureFit resolutions: "
            f"groups={len(program.recomputation_groups)}, "
            f"selections={len(resolved)}"
        )

    started = time.perf_counter_ns()
    # One bound for the whole search. Each resolved program is planned
    # against what the ones before it admitted, which is what makes their
    # order worth choosing.
    shared = best if best is not None else BestFound()

    def evaluate(
        resolution: Resolution,
    ) -> tuple[SelectionProblem, CProblemResult | None]:
        return evaluate_resolution(
            program,
            resolution=resolution,
            initial_residency=initial_residency,
            final_residency=final_residency,
            config=config,
            options=selected_options,
            admission=admission,
            best=shared,
            progress=progress,
        )

    if worker_count(selected_options, len(resolved)) == 1:
        evaluated = tuple(evaluate(resolution) for resolution in resolved)
    else:
        # Dispatched together so that every resolved program's units reach
        # the shared pool as one batch. Evaluating them one after another
        # instead costs a barrier per resolution, and the pool spends most of
        # its width idle waiting for the slowest unit of a five-unit round.
        #
        # These threads only wait: `run_problems` submits its units to the
        # shared pool and blocks, so the compiled work stays bounded by that
        # pool rather than by the number of resolutions. They must not come
        # from the shared pool itself, or waiting on it from inside it could
        # starve the units they are waiting for.
        with ThreadPoolExecutor(max_workers=len(resolved)) as executor:
            # `map` yields in argument order, so the result order -- and
            # every tie-break that depends on it -- does not depend on which
            # resolution finished first.
            evaluated = tuple(executor.map(evaluate, resolved))
    valid = tuple(
        (problem, result) for problem, result in evaluated if result is not None
    )
    if progress is not None:
        progress(
            "PressureFit compiled problems and candidates finished: "
            f"valid={len(valid)}/{len(evaluated)}, "
            "candidates="
            f"{sum(len(result.candidates) for _problem, result in valid)}, "
            f"workers={worker_count(selected_options, len(evaluated))}, "
            f"elapsed={(time.perf_counter_ns() - started) / 1e9:.3f}s"
        )
    if not valid:
        raise RuntimeError(
            "PressureFit rejected every selection after semantic "
            "feasibility validation succeeded"
        )
    # One decode across every resolved program, so the winner and the
    # diagnostics are exactly what a single batched evaluation produced.
    return finish_pressurefit(
        program,
        initial_residency,
        final_residency,
        config,
        selected_options,
        tuple(problem for problem, _result in valid),
        tuple(result for _problem, result in valid),
        admission,
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
        resolutions=resolutions(program),
        progress=None,
    )
    preflight_problems(problems)


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
            result = plan_program(
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


__all__ = [
    "ordered_resolutions",
    "plan_program",
    "pressurefit",
    "validate_schedule_feasibility",
]
