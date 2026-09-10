from __future__ import annotations

from dataclasses import replace

import pytest

from reference.python.pressurefit.facts import build_facts
from reference.python.pressurefit.residency import (
    ResidencyPlan,
    Span,
    _pressure_by_device,
    _required_floor_pressure,
    boundary_bytes,
    extend_interval_entries,
    reduce_pressure,
    seed_residency,
)
from shadowspill.planner import GenericPlanningOptions, pressurefit
from shadowspill.planner.request import InitialPlacement

from ...._examples import (
    config,
    exact_capacity_program,
    exact_capacity_residency,
    training_chain_config,
    training_chain_initial,
    training_chain_program,
)


@pytest.mark.parametrize("workers", [1, 2, 0])
def test_candidate_parallelism_preserves_the_complete_result(workers: int) -> None:
    initial, final = exact_capacity_residency()
    options = GenericPlanningOptions(minimum_object_bytes_evict_eligible=0)

    result = pressurefit(
        exact_capacity_program(),
        initial_residency=initial,
        final_residency=final,
        config=config(),
        generic=options,
        workers=workers,
    )

    assert (
        result.schedule.digest
        == "c530d01e90dab80c7396fcc61e679341df07b746310c95f8e885d94ffd512e30"
    )
    assert result.diagnostics.selected_makespan_ns == 5_000
    # two strategies, four rules, two coalescing modes
    assert result.diagnostics.candidate_evaluation_count == 16


def test_names_do_not_affect_schedule_geometry_or_makespan() -> None:
    program = exact_capacity_program()
    renamed = replace(
        program,
        profiles=(replace(program.profiles[0], compatibility_digest="other_abi"),),
    )
    initial, final = exact_capacity_residency()

    original = pressurefit(
        program,
        initial_residency=initial,
        final_residency=final,
        config=config(),
        generic=GenericPlanningOptions(minimum_object_bytes_evict_eligible=0),
        workers=1,
    )
    other = pressurefit(
        renamed,
        initial_residency=initial,
        final_residency=final,
        config=config(),
        generic=GenericPlanningOptions(minimum_object_bytes_evict_eligible=0),
        workers=1,
    )

    assert original.schedule == other.schedule
    assert original.simulation.makespan_ns == other.simulation.makespan_ns


@pytest.mark.parametrize("fetch_headroom", [False, True])
def test_pressure_sweep_matches_scalar_boundaries(
    fetch_headroom: bool,
) -> None:
    program = training_chain_program(10)
    initial = training_chain_initial(10)
    selected_config = training_chain_config(500)
    facts = build_facts(program, (), initial, (), selected_config)
    seed = seed_residency(facts, selected_config, InitialPlacement.GREEDY)
    plan = reduce_pressure(facts, selected_config, seed, "tight-stall")

    swept = _pressure_by_device(
        facts,
        plan,
        fetch_headroom=fetch_headroom,
    )
    for device_id in facts.object_capacity_by_device:
        assert swept[device_id] == tuple(
            boundary_bytes(
                facts,
                plan,
                boundary,
                device_id,
                fetch_headroom=fetch_headroom,
            )
            for boundary in range(-1, facts.last_boundary + 1)
        )


def test_direct_required_floor_matches_minimal_residency_sweep() -> None:
    program = training_chain_program(10)
    initial = training_chain_initial(10)
    selected_config = training_chain_config(500)
    facts = build_facts(program, (), initial, (), selected_config)
    minimal = ResidencyPlan(
        tuple(
            tuple(Span(value, value) for value in sorted(anchors))
            for anchors in facts.anchors
        ),
        facts.anchors,
    )

    assert _required_floor_pressure(facts) == _pressure_by_device(facts, minimal)


def test_interval_extension_matches_scalar_admission() -> None:
    program = training_chain_program(10)
    initial = training_chain_initial(10)
    selected_config = training_chain_config(500)
    facts = build_facts(program, (), initial, (), selected_config)
    seed = seed_residency(facts, selected_config, InitialPlacement.GREEDY)
    plan = reduce_pressure(facts, selected_config, seed, "tight-stall")

    scalar = plan
    for alias in range(len(facts.alias_ids)):
        span_index = 1
        while span_index < len(scalar.spans[alias]):
            span = scalar.spans[alias][span_index]
            previous = scalar.spans[alias][span_index - 1]
            candidate_start = span.start - 1
            if candidate_start <= previous.end:
                span_index += 1
                continue
            spans = list(scalar.spans)
            alias_spans = list(spans[alias])
            alias_spans[span_index] = Span(candidate_start, span.end)
            spans[alias] = tuple(alias_spans)
            proposed = ResidencyPlan(tuple(spans), scalar.anchors)
            device_id = facts.alias_devices[alias]
            if (
                boundary_bytes(facts, proposed, candidate_start, device_id)
                <= facts.object_capacity_by_boundary[device_id][candidate_start + 1]
            ):
                scalar = proposed
                continue
            span_index += 1

    assert extend_interval_entries(facts, plan) == scalar


# These plan under whatever `PressureFitOptions` defaults to, so the default's
# initial placement is baked into every value here. They were re-frozen when that
# default became `REQUIRED`: the makespans rose by 4 to 14 microseconds and the
# action counts by one to eight, because required places only what the
# declaration and the first task demand and fetches the rest on the schedule,
# where the simulator charges for it. Greedy's lower numbers were not a faster
# plan -- they were the same work with the opening restore left out of the
# makespan. The selected candidate is unchanged at every fixture.
@pytest.mark.parametrize(
    ("layers", "capacity", "digest", "makespan_ns", "candidate", "actions"),
    (
        (
            1,
            224,
            "0ffa4e2af838ff5cc44c3a3bdc18d0d2b4e1fe7cdb616c15829a8f466a45e727",
            64_000,
            "tight-stall/packed-fit",
            14,
        ),
        (
            2,
            224,
            "de9fbf3f76897c59937976c8220c7531a929fbe19e878210b01020fc54e2d2f1",
            114_000,
            "tight-stall/packed-fit",
            25,
        ),
        (
            5,
            800,
            "a4696af393b24fb71d0a9ba2dbc9630e1b52c7fa762b9ad7dd1326826571c332",
            166_000,
            "headroom-stall/packed-fifo",
            40,
        ),
        (
            10,
            500,
            "96dcbfe6b4065c585e60396c7ece819d9f6e20f9cb70fc77505f0a0d2aea0405",
            310_000,
            "headroom-stall/packed-fifo",
            103,
        ),
    ),
)
def test_training_chain_schedule_artifacts_are_frozen(
    layers: int,
    capacity: int,
    digest: str,
    makespan_ns: int,
    candidate: str,
    actions: int,
) -> None:
    result = pressurefit(
        training_chain_program(layers),
        initial_residency=training_chain_initial(layers),
        config=training_chain_config(capacity),
        generic=GenericPlanningOptions(minimum_object_bytes_evict_eligible=0),
        workers=1,
    )

    assert result.schedule.digest == digest
    assert result.simulation.makespan_ns == makespan_ns
    assert result.diagnostics.selected_candidate_id == candidate
    assert len(result.schedule.actions) == actions
