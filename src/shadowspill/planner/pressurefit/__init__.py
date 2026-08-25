"""PressureFit: the best schedule for one Program under one budget.

The work divides four ways, one module each:

``recomputation``  which recomputation selections are worth evaluating
``search``         evaluating them and picking a winner
``refinement``     what to try when admission refuses that winner
this module        the entry point, and validating what it was given
"""

from __future__ import annotations

from collections.abc import Callable

from shadowspill.ir import (
    Program,
    ResidencySpec,
    shared_residency_footprint,
)
from shadowspill.simulator import SimulationConfig
from shadowspill.simulator.capi import simulator_api

from ..admission import AdmissionFacts
from ..best import BestPlaced
from ..capi import planner_api
from ..recomputation import Resolution
from ..request import PressureFitOptions
from .candidates import CProblemResult
from .search import (
    SelectionProblem,
    build_problems,
    preflight_problems,
    run_problems,
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


def evaluate_resolution(
    program: Program,
    *,
    resolution: Resolution,
    initial_residency: tuple[ResidencySpec, ...],
    final_residency: tuple[ResidencySpec, ...] = (),
    config: SimulationConfig,
    options: PressureFitOptions,
    admission: AdmissionFacts | None = None,
    placement: AdmissionFacts | None = None,
    best: BestPlaced | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[SelectionProblem, CProblemResult | None]:
    """Plan one resolved program: every candidate policy, one task set.

    This is PressureFit proper. It receives a task set with every
    alternative already fixed and knows nothing about what was chosen
    between, or that anything was. The caller decides which resolved
    programs exist and in what order they are tried.

    Returns the compiled problem beside its result so the caller can decode
    a winner across several resolved programs at once; `None` means this
    resolution was rejected before any candidate ran.
    """

    simulator_api()
    planner_api()
    problems = preflight_problems(
        build_problems(
            program,
            initial_residency,
            final_residency,
            config,
            admission,
            placement=placement,
            resolutions=(resolution,),
            progress=progress,
        )
    )
    results = run_problems(problems, options, best=best)
    # The candidates publish to the record themselves, as they place, so
    # there is nothing to offer here: by the time this returns, anything
    # worth sharing is already shared.
    return problems[0], results[0]


__all__ = ["evaluate_resolution", "validate_pressurefit_inputs"]
