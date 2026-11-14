"""Quantized fetch triggers retain the next consumer's priority."""

from __future__ import annotations

from dataclasses import replace

import pytest

from reference.python.pressurefit import pressurefit as reference_pressurefit
from shadowspill.ir import (
    AliasGroupSpec,
    MemoryActionKind,
    MemoryLocation,
    ObjectSpec,
    ResidencySpec,
    ShadowSpillProgram,
    TaskProfile,
    TaskSpec,
)
from shadowspill.planner import GenericPlanningOptions
from shadowspill.planner.search.algorithms.pressurefit import PressureFit
from shadowspill.planner.search.algorithms.pressurefit.options import PressureFitOptions
from shadowspill.simulator import simulate

from ...._examples import COMPUTE, DEVICE, config


@pytest.mark.parametrize("rule", ["latest-safe", "packed-fifo", "packed-fit"])
def test_same_trigger_fetches_follow_consumer_order_and_avoid_exposed_stall(rule):
    # Alias order opposes consumption order, as it does for gradients
    # produced in ascending microbatch order and consumed in reverse.
    program = ShadowSpillProgram(
        devices=(DEVICE,),
        alias_groups=(
            AliasGroupSpec("later", "cuda_0", 61),
            AliasGroupSpec("sooner", "cuda_0", 61),
        ),
        objects=(
            ObjectSpec("later_value", "later", 0, 61),
            ObjectSpec("sooner_value", "sooner", 0, 61),
        ),
        profiles=(
            TaskProfile("control", 0, 0, "task_abi"),
            TaskProfile("compute", 1_000, 0, "task_abi"),
        ),
        tasks=(
            TaskSpec("setup", COMPUTE, "control"),
            TaskSpec("first", COMPUTE, "compute", inputs=("sooner_value",)),
            TaskSpec("second", COMPUTE, "compute", inputs=("later_value",)),
        ),
    )
    arguments = dict(
        initial_residency=tuple(
            ResidencySpec(a.alias_group_id, MemoryLocation.SPILL)
            for a in program.alias_groups
        ),
        config=config(),
        generic=GenericPlanningOptions(minimum_object_bytes_evict_eligible=0),
        workers=1,
    )
    options = PressureFitOptions(
        residency_strategies=("tight-stall",), fetch_rules=(rule,)
    )
    result = PressureFit(options)(program, **arguments)
    reference = reference_pressurefit(program, algorithm_options=options, **arguments)
    assert result.schedule == reference.schedule
    assert result.simulation == reference.simulation
    fetches = [a for a in result.schedule.actions if a.kind is MemoryActionKind.FETCH]
    assert [(a.trigger_task_id, a.alias_group_id) for a in fetches] == [
        ("setup", "sooner"),
        ("setup", "later"),
    ]
    assert result.simulation.makespan_ns == 3_000
    first = next(t for t in result.simulation.task_intervals if t.task_id == "first")
    assert first.start_ns == 1_000
    # Counterfactual changes only the same-boundary FIFO ordering. The
    # later-needed copy otherwise adds a whole transfer to the critical path.
    indices = [
        i
        for i, a in enumerate(result.schedule.actions)
        if a.kind is MemoryActionKind.FETCH
    ]
    old = list(result.schedule.actions)
    old[indices[0]], old[indices[1]] = old[indices[1]], old[indices[0]]
    delayed = simulate(
        program, replace(result.schedule, actions=tuple(old)), config=config()
    )
    assert delayed.makespan_ns == 4_000
