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
from shadowspill.planner import PressureFitOptions, pressurefit
from shadowspill.planner.request import InitialPlacement

from ._examples import (
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
    options = PressureFitOptions(workers=workers, minimum_object_bytes_evict_eligible=0)

    result = pressurefit(
        exact_capacity_program(),
        initial_residency=initial,
        final_residency=final,
        config=config(),
        options=options,
    )

    assert (
        result.schedule.digest
        == "c530d01e90dab80c7396fcc61e679341df07b746310c95f8e885d94ffd512e30"
    )
    assert result.diagnostics.selected_makespan_ns == 5_000
    assert result.diagnostics.candidate_evaluation_count == 32


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
        options=PressureFitOptions(minimum_object_bytes_evict_eligible=0),
    )
    other = pressurefit(
        renamed,
        initial_residency=initial,
        final_residency=final,
        config=config(),
        options=PressureFitOptions(minimum_object_bytes_evict_eligible=0),
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


@pytest.mark.parametrize(
    ("layers", "capacity", "digest", "makespan_ns", "candidate", "actions"),
    (
        (
            1,
            224,
            "9f604b7bd5be7b026dcfb7cecf1ac4053f0c4d5951b8eacc8b17352d12a45dea",
            56_000,
            "tight-stall/packed-fit",
            13,
        ),
        (
            2,
            224,
            "45c58b17c637fa33f978d09e7f14cd94bbecf52456b92f3731c8b77b92da6724",
            110_000,
            "tight-stall/packed-fit",
            24,
        ),
        (
            5,
            800,
            "0ea05b8eeba4125a5a86e7d54df85165979e838d8ed8963a46db301e215e774f",
            152_000,
            "headroom-stall/packed-fifo",
            32,
        ),
        (
            10,
            500,
            "995bd7dba563fcb7c1e402c524afdca25f105fdfce46f9a2e2bae4d32b0aec44",
            302_000,
            "headroom-stall/packed-fifo",
            98,
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
        options=PressureFitOptions(minimum_object_bytes_evict_eligible=0),
    )

    assert result.schedule.digest == digest
    assert result.simulation.makespan_ns == makespan_ns
    assert result.diagnostics.selected_candidate_id == candidate
    assert len(result.schedule.actions) == actions
