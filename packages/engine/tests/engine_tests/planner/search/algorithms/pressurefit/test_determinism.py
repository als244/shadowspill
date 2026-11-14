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
    reduce_pressure,
    seed_residency,
)
from shadowspill.planner import GenericPlanningOptions, pressurefit

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
        == "f7443cc16b764df6e26ae980807157ff6d7f6bb12c28139ee048ea19d4322060"
    )
    assert result.diagnostics.selected_makespan_ns == 4_000
    # Two residency strategies and four fetch rules.
    assert result.diagnostics.candidate_evaluation_count == 8


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
    seed = seed_residency(facts)
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


@pytest.mark.parametrize(
    ("layers", "capacity", "digest", "makespan_ns", "candidate", "actions"),
    (
        (
            1,
            224,
            "2b2c8a40cd0bf39928f6e58fd0027b105be9a406f22797ef141f60a2b08b327a",
            64_000,
            "tight-stall/packed-fit",
            15,
        ),
        (
            2,
            224,
            "8930d44dacbbcab9e299071ee903033cd084c9c02e7d73166c69cb4c57b7766c",
            118_000,
            "tight-stall/packed-fit",
            26,
        ),
        (
            5,
            800,
            "90f81d896f3470ac31857ae278365418352cc36e2aab4974fe717386de81f9d6",
            164_000,
            "headroom-stall/packed-fifo",
            41,
        ),
        (
            10,
            500,
            "7d4b160615a1fa712d13437dd5246d947d00b7a71bc7ce287077a9bc1031c54d",
            310_000,
            "headroom-stall/packed-fifo",
            104,
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
