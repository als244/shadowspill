"""PressureFit: the search that ships.

One implementation of :class:`~shadowspill.planning.memory_search.SearchAlgorithm`.
Given a program, where it must start and end, and a machine, it answers with the
best schedule it can find under that machine's capacity.

The work divides up:

``options``        what it may try, and how hard
``capi``           its own ctypes surface, mirroring its public C header
``candidates``     the library call and the C option structs
``best``           the best plan placed so far, shared across a search
``search``         building problems, running them, decoding a winner
this module        the search object, expansion, and validating its inputs

Expanding a program into resolved programs is done here rather than by the
planner: a program arrives with its task alternatives still open, and how
many ways to fix them are worth planning -- and in what order -- is a
search's own judgement. The order matters because a plan admitted under any
resolved program bounds the search under every other one.

The algorithm itself is described in ``docs/architecture/pressurefit.md``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace
from fractions import Fraction

from shadowspill.errors import PlanInfeasibleError
from shadowspill.ir import (
    ResidencySpec,
    ShadowSpillProgram,
)
from shadowspill.planning.memory_search.admission import AdmissionFacts
from shadowspill.planning.memory_search.algorithm import (
    SearchAlgorithm,
    SearchOptions,
    set_default_algorithm,
)
from shadowspill.planning.memory_search.request import GenericPlanningOptions
from shadowspill.planning.memory_search.result import ProgramPlanResult
from shadowspill.planning.memory_search.toolkit.resolution import (
    Resolution,
    resolutions,
)
from shadowspill.planning.memory_search.toolkit.validation import validate_search_inputs
from shadowspill.simulator import SimulationConfig
from shadowspill.simulator.capi import simulator_api

from .best import BestPlaced
from .candidates import (
    CProblemResult,
)
from .capi import (
    pressurefit_api,
)
from .options import (
    PressureFitOptions,
)
from .search import (
    SelectionProblem,
    build_problems,
    finish_pressurefit,
    preflight_problems,
    run_problems,
)

#: What a group's alternatives are called. Ordering only needs to
#: recognise the two extremes; anything else falls through to the middle.
_RECOMPUTE = "recompute"
_SAVE = "save"


def _recompute_share(resolution: Resolution) -> float:
    """Fraction of this resolution's groups that recompute."""

    if not resolution:
        return 0.0
    recomputed = sum(1 for item in resolution if item.option_id == _RECOMPUTE)
    return recomputed / len(resolution)


def _evaluate_resolutions(
    program: ShadowSpillProgram,
    *,
    resolutions: tuple[Resolution, ...],
    initial_residency: tuple[ResidencySpec, ...],
    final_residency: tuple[ResidencySpec, ...] = (),
    config: SimulationConfig,
    generic: GenericPlanningOptions,
    algorithm: PressureFitOptions,
    workers: int = 0,
    admission: AdmissionFacts | None = None,
    placement: AdmissionFacts | None = None,
    best: BestPlaced | None = None,
    progress: Callable[[str], None] | None = None,
    incumbent: ProgramPlanResult | None = None,
) -> tuple[tuple[SelectionProblem, CProblemResult | None], ...]:
    """Plan every resolved program: each one's candidate policies, one call.

    This is PressureFit proper. It receives task sets with every alternative
    already fixed and knows nothing about what was chosen between them. The
    caller decides which resolved programs exist and in what order.

    They are evaluated together because that is what shares the placement
    record between them, and because the library's workers are then free to
    move between resolved programs rather than idling on the slowest one.

    Returns each compiled problem beside its result, so the caller can decode
    a winner across all of them at once; a `None` result means that resolved
    program was rejected before any candidate ran.
    """

    simulator_api()
    pressurefit_api()
    problems = preflight_problems(
        build_problems(
            program,
            initial_residency,
            final_residency,
            config,
            admission,
            placement=placement,
            resolutions=resolutions,
            progress=progress,
            incumbent=incumbent,
        )
    )
    results = run_problems(problems, generic, algorithm, workers=workers, best=best)
    # The candidates publish to the record themselves, as they place, so
    # there is nothing to offer here: by the time this returns, anything
    # worth sharing is already shared.
    return tuple(zip(problems, results, strict=True))


def ordered_resolutions(
    program: ShadowSpillProgram,
    resolution_options: tuple[Fraction, ...],
    tail_save_counts: tuple[int, ...] = (0,),
) -> tuple[Resolution, ...]:
    """Return the resolved programs to try, most recomputed first.

    Order is part of the algorithm, not a detail of it. A plan placed under
    any resolved program bounds the search under every later one, so the order
    decides how much work the search does — and, more sharply, whether the
    bound exists early enough to prevent any work at all.

    The rule is one sort: descending share of groups recomputed. Recomputing
    frees the memory that is binding under pressure, so a more-recomputed
    resolution is both likelier to place a plan at all and likelier to be the
    one that wins.
    """

    resolved = resolutions(
        program, resolution_options, tail_save_counts=tail_save_counts
    )
    if len(resolved) < 2:
        return resolved
    ranked = sorted(
        enumerate(resolved),
        key=lambda pair: (-_recompute_share(pair[1]), pair[0]),
    )
    return tuple(resolution for _position, resolution in ranked)


class PressureFit(SearchAlgorithm):
    """The search that ships, built with the options it searches under.

    A `SearchAlgorithm` named ``"pressurefit"``. It expands a program's task
    alternatives into the resolved programs its options name, searches each
    one with every residency strategy crossed with every fetch rule, and
    answers with the fastest plan any of them reached.

    Parameters
    ----------
    options : PressureFitOptions, optional
        What it may try, and how hard. None uses ``PressureFitOptions()``.

    Raises
    ------
    TypeError
        If ``options`` is not a `PressureFitOptions`.
    """

    name = "pressurefit"

    def __init__(self, options: PressureFitOptions | None = None) -> None:
        if options is not None and not isinstance(options, PressureFitOptions):
            raise TypeError(
                f"PressureFit takes PressureFitOptions, not {type(options).__name__}"
            )
        self.options: PressureFitOptions = (
            options if options is not None else PressureFitOptions()
        )

    def preflight(
        self,
        program: ShadowSpillProgram,
        *,
        initial_residency: tuple[ResidencySpec, ...],
        final_residency: tuple[ResidencySpec, ...] = (),
        config: SimulationConfig,
        admission: AdmissionFacts | None = None,
        generic: GenericPlanningOptions,
    ) -> None:
        """Refuse a machine on which no resolved program of ``program`` fits.

        Checks each resolved program the options name: at every task, its
        workspace and the inputs and outputs it needs must fit the device
        together, and every input must be able to reach the device by its
        first use. A search plans only the resolved programs that pass.

        Parameters
        ----------
        program : ShadowSpillProgram
            As in `SearchAlgorithm.preflight`.
        initial_residency : tuple of ResidencySpec
            As in `SearchAlgorithm.preflight`.
        final_residency : tuple of ResidencySpec, default ()
            As in `SearchAlgorithm.preflight`.
        config : SimulationConfig
            As in `SearchAlgorithm.preflight`.
        admission : AdmissionFacts, optional
            As in `SearchAlgorithm.preflight`.
        generic : GenericPlanningOptions
            Not read.

        Raises
        ------
        TypeError
            If an argument has the wrong type.
        ValueError
            If ``admission`` does not describe ``program``, or its object
            capacity disagrees with ``config``.
        PlanInfeasibleError
            If no resolved program passes. The error is the first one's.
        """

        validate_search_inputs(
            program, initial_residency, final_residency, config, admission
        )
        simulator_api()
        pressurefit_api()
        preflight_problems(
            build_problems(
                program,
                initial_residency,
                final_residency,
                config,
                admission,
                resolutions=ordered_resolutions(
                    program,
                    self.options.resolution_options,
                    self.options.tail_save_counts,
                ),
                progress=None,
            )
        )

    def record_trace(
        self,
        result: ProgramPlanResult,
        *,
        options: SearchOptions,
        admission: AdmissionFacts | None = None,
        placement: AdmissionFacts | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        """Record the cut history of each traced candidate, rerunning it alone.

        Each call makes a new ``pressurefit-*`` directory under
        ``options.trace_directory`` holding ``manifest.json``,
        ``program.json`` and one ``.pftrace`` file per candidate. Each
        candidate is rerun alone on one worker and checked against its
        original diagnostic. The manifest ends ``"complete"``, or
        ``"failed"`` with the error.

        Parameters
        ----------
        result : ProgramPlanResult
            As in `SearchAlgorithm.record_trace`.
        options : SearchOptions
            As in `SearchAlgorithm.record_trace`. Its generic options must be
            ``deterministic``.
        admission : AdmissionFacts, optional
            As in `SearchAlgorithm.record_trace`.
        placement : AdmissionFacts, optional
            As in `SearchAlgorithm.record_trace`.
        progress : callable, optional
            Receives one line per traced candidate, then the directory.

        Raises
        ------
        ValueError
            If ``options`` names no ``trace_directory`` or is not
            ``deterministic``; the plan is for more than one device; a pair to
            trace was not evaluated; or no pairs are named and the answer is
            the plan to beat, which has no cut history in this search.
        TypeError
            If ``options`` names a search whose options are not
            `PressureFitOptions`.
        RuntimeError
            If a rerun differs from its original candidate.
        """
        from .tracing import (
            record_plan_trace,
        )

        record_plan_trace(
            result,
            options=options,
            admission=admission,
            placement=placement,
            progress=progress,
        )

    def __call__(
        self,
        program: ShadowSpillProgram,
        *,
        initial_residency: tuple[ResidencySpec, ...],
        final_residency: tuple[ResidencySpec, ...] = (),
        config: SimulationConfig,
        generic: GenericPlanningOptions,
        workers: int = 0,
        admission: AdmissionFacts | None = None,
        placement: AdmissionFacts | None = None,
        progress: Callable[[str], None] | None = None,
        incumbent: ProgramPlanResult | None = None,
        best: BestPlaced | None = None,
        keep_resolutions: bool = False,
    ) -> ProgramPlanResult:
        """Select a schedule for `program`, planning each resolution in turn.

        This search's own `options` name, among other things, which resolved
        programs to plan. `best` carries a plan already in hand across
        resolved programs, so each one is searched against the answer the
        previous ones found; omitting it means this call starts from nothing
        and keeps its own.

        Capacity is settled inside the search: a candidate measures its own
        plan against the pool `placement` describes and gives capacity back
        until the plan fits, so there is nothing to retry at this level.

        `incumbent` is the plan to beat: a result for this same program
        found elsewhere -- at a smaller capacity, say. It is measured at
        this capacity before any candidate runs, so a candidate that cannot
        beat it is never measured, and it is answered with unless a
        candidate does strictly better. It reaches the resolved program it
        was found for; a search over resolution options that do not include
        that one carries none.

        `keep_resolutions` materialises every resolved program's selected
        schedule the way the answer's is, and reports them beside it.
        """

        validate_search_inputs(
            program, initial_residency, final_residency, config, admission
        )
        if incumbent is not None and incumbent.program.digest != program.digest:
            raise ValueError("the plan to beat is a plan for a different program")
        resolved = ordered_resolutions(
            program, self.options.resolution_options, self.options.tail_save_counts
        )
        if progress is not None:
            progress(
                "PressureFit resolutions: "
                f"groups={len(program.task_alternative_groups)}, "
                f"selections={len(resolved)}"
            )

        started = time.perf_counter_ns()
        # One record for the whole search. Every candidate under every
        # resolved program measures against what has already been placed,
        # which is what makes measuring affordable and what makes their
        # order worth choosing.
        owned = BestPlaced() if best is None else None
        shared = best if best is not None else owned
        try:
            result = self._evaluate(
                program,
                resolved=resolved,
                initial_residency=initial_residency,
                final_residency=final_residency,
                config=config,
                search_options=SearchOptions(
                    generic=generic, algorithm=self, workers=workers
                ),
                workers=workers,
                admission=admission,
                placement=placement,
                shared=shared,
                progress=progress,
                incumbent=incumbent,
                started=started,
                keep_resolutions=keep_resolutions,
            )
        finally:
            if owned is not None:
                owned.close()
        return replace(
            result,
            diagnostics=replace(
                result.diagnostics,
                effective_object_capacity_bytes=(
                    None if admission is None else admission.object_capacity_bytes
                ),
                workers=workers,
            ),
            admission_facts=admission,
        )

    def _evaluate(
        self,
        program: ShadowSpillProgram,
        *,
        resolved: tuple[Resolution, ...],
        initial_residency: tuple[ResidencySpec, ...],
        final_residency: tuple[ResidencySpec, ...],
        config: SimulationConfig,
        search_options: SearchOptions,
        workers: int,
        admission: AdmissionFacts | None,
        placement: AdmissionFacts | None,
        shared: BestPlaced | None,
        progress: Callable[[str], None] | None,
        incumbent: ProgramPlanResult | None,
        started: int,
        keep_resolutions: bool = False,
    ) -> ProgramPlanResult:
        """Every resolved program, evaluated in one call, then one decode.

        A resolved program that cannot satisfy the semantic-capacity
        preflight is not an answer about the program: it says this one way
        of fixing the save/recompute alternatives does not fit, which is
        exactly the question this layer exists to ask several times. A
        rejection is therefore filtered here, and only a program with no
        viable resolution at all is infeasible.

        Threads belong to the library and to this call. Handing it every
        resolved program at once is what lets a worker move between them
        instead of idling on the slowest, and what shares the placement
        record across them.
        """

        evaluated = _evaluate_resolutions(
            program,
            resolutions=resolved,
            initial_residency=initial_residency,
            final_residency=final_residency,
            config=config,
            generic=search_options.generic,
            algorithm=self.options,
            workers=workers,
            admission=admission,
            placement=placement,
            best=shared,
            progress=progress,
            incumbent=incumbent,
        )
        valid = tuple(
            (problem, result) for problem, result in evaluated if result is not None
        )
        if progress is not None:
            progress(
                "PressureFit compiled problems and candidates finished: "
                f"valid={len(valid)}/{len(evaluated)}, "
                "candidates="
                f"{sum(len(result.candidates) for _problem, result in valid)}, "
                f"workers={workers or 'auto'}, "
                f"elapsed={(time.perf_counter_ns() - started) / 1e9:.3f}s"
            )
        if not valid:
            # Every resolved program failed the planner's own analytic
            # capacity check: nothing fits at this capacity, and the caller
            # hears that as infeasibility, as it would from any one of them.
            raise PlanInfeasibleError(
                "every resolution of the program is analytically infeasible "
                "at this capacity",
                kind="analytic_capacity",
            )
        # One decode across every resolved program, so the winner and the
        # diagnostics are exactly what a single batched evaluation produced.
        return finish_pressurefit(
            program,
            initial_residency,
            final_residency,
            config,
            search_options,
            tuple(problem for problem, _result in valid),
            tuple(result for _problem, result in valid),
            admission,
            shared,
            placement=placement,
            incumbent=incumbent,
            keep_resolutions=keep_resolutions,
        )


#: The search that ships: a `PressureFit` built with the default
#: `PressureFitOptions`. `SearchOptions` runs it when ``algorithm`` is None.
pressurefit = PressureFit()

# What a planning call gets when it names no algorithm.
set_default_algorithm(pressurefit)

__all__ = [
    "PressureFit",
    "PressureFitOptions",
    "ordered_resolutions",
    "pressurefit",
]
