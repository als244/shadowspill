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
from dataclasses import replace

from shadowspill.ir import Program, ResidencySpec
from shadowspill.simulator import SimulationConfig
from shadowspill.simulator.capi import simulator_api

from .admission import AdmissionFacts
from .best import BestPlaced
from .capi import planner_api
from .pressurefit import evaluate_resolutions, validate_pressurefit_inputs
from .pressurefit.search import (
    build_problems,
    finish_pressurefit,
    preflight_problems,
)
from .recomputation import Resolution, resolutions
from .request import PressureFitOptions
from .result import PressureFitResult

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
    """Return the resolved programs to try, most recomputed first.

    Order is part of the algorithm, not a detail of it. A plan placed under
    any resolved program bounds the search under every later one, so the order
    decides how much work the search does — and, more sharply, whether the
    bound exists early enough to prevent any work at all.

    The rule is one sort: descending share of groups recomputed. Recomputing
    frees the memory that is binding under pressure, so a more-recomputed
    resolution is both likelier to place a plan at all and likelier to be the
    one that wins. Measured across the 2,520-point corpus, win rate follows
    that share without exception:

        recomputed   97%    74%    49%    25%     0%
        wins        68.5%  17.7%   8.7%   4.3%   2.7%

    So position in this order is roughly how likely a resolution is to be the
    answer, which is exactly what a search wants to try first: the strongest
    bound arrives soonest, and everything after it searches against a real
    plan rather than an empty record.

    "Frees memory" is the likely case rather than a guarantee: peak working
    space rises when recomputing, so a program could in principle fail on
    workspace at full recompute while fitting with some saved. This is a
    heuristic about where to look first and never a claim about what is
    feasible, so being wrong costs a little work and nothing else.

    Ties keep the order the program listed them in, so the result is a
    function of the program alone.
    """

    resolved = resolutions(program)
    if len(resolved) < 2:
        return resolved
    ranked = sorted(
        enumerate(resolved),
        key=lambda pair: (-_recompute_share(pair[1]), pair[0]),
    )
    return tuple(resolution for _position, resolution in ranked)


def plan_program(
    program: Program,
    *,
    initial_residency: tuple[ResidencySpec, ...],
    final_residency: tuple[ResidencySpec, ...] = (),
    config: SimulationConfig,
    options: PressureFitOptions | None = None,
    admission: AdmissionFacts | None = None,
    placement: AdmissionFacts | None = None,
    best: BestPlaced | None = None,
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
    # One record for the whole search. Every candidate under every resolved
    # program measures against what has already been placed, which is what
    # makes measuring affordable and what makes their order worth choosing.
    owned = BestPlaced() if best is None else None
    shared = best if best is not None else owned
    try:
        # Every resolved program, evaluated in one call.
        #
        # A resolved program that cannot satisfy the semantic-capacity
        # preflight is not an answer about the Program: it says this one way
        # of fixing the save/recompute alternatives does not fit, which is
        # exactly the question this layer exists to ask several times. Under
        # pressure the least-recomputed resolution routinely fails it while
        # the recompute-heavy ones admit, so a rejection is filtered there,
        # and only a Program with no viable resolution at all is infeasible.
        #
        # Threads belong to the library and to this call. Handing it every
        # resolved program at once is what lets a worker move between them
        # instead of idling on the slowest, and what shares the placement
        # record across them.
        try:
            evaluated = evaluate_resolutions(
                program,
                resolutions=resolved,
                initial_residency=initial_residency,
                final_residency=final_residency,
                config=config,
                options=selected_options,
                admission=admission,
                placement=placement,
                best=shared,
                progress=progress,
            )
        except ValueError:
            # Every resolved program was rejected, so the Program itself has
            # no viable way to fix its alternatives.
            raise
        valid = tuple(
            (problem, result) for problem, result in evaluated if result is not None
        )
        if progress is not None:
            progress(
                "PressureFit compiled problems and candidates finished: "
                f"valid={len(valid)}/{len(evaluated)}, "
                "candidates="
                f"{sum(len(result.candidates) for _problem, result in valid)}, "
                f"workers={selected_options.workers or 'auto'}, "
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
            shared,
        )
    finally:
        if owned is not None:
            owned.close()


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
    placement: AdmissionFacts | None = None,
    progress: Callable[[str], None] | None = None,
) -> PressureFitResult:
    """Select a schedule for `program`.

    Capacity is settled inside the search: a candidate measures its own
    plan against the pool `placement` describes and gives capacity back
    until the plan fits, so there is nothing to retry at this level.
    """

    result = plan_program(
        program,
        initial_residency=initial_residency,
        final_residency=final_residency,
        config=config,
        options=options,
        admission=admission,
        placement=placement,
        progress=progress,
    )
    return replace(
        result,
        diagnostics=replace(
            result.diagnostics,
            effective_object_capacity_bytes=(
                None if admission is None else admission.object_capacity_bytes
            ),
        ),
        admission_facts=admission,
    )


__all__ = [
    "ordered_resolutions",
    "plan_program",
    "pressurefit",
    "validate_schedule_feasibility",
]
