from __future__ import annotations

from dataclasses import replace

import pytest

from shadowspill.planner import PressureFitOptions, pressurefit

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
    options = PressureFitOptions(workers=workers)

    result = pressurefit(
        exact_capacity_program(),
        initial_residency=initial,
        final_residency=final,
        config=config(),
        options=options,
    )

    assert (
        result.schedule.digest
        == "39a5428ad479d2a19a6a74e6b4f13472d62c25805c6bd91b8ff62ad45d0d055e"
    )
    assert result.diagnostics.selected_makespan_ns == 5_000
    assert result.diagnostics.candidate_count == 40


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
    )
    other = pressurefit(
        renamed,
        initial_residency=initial,
        final_residency=final,
        config=config(),
    )

    assert original.schedule == other.schedule
    assert original.simulation.makespan_ns == other.simulation.makespan_ns


@pytest.mark.parametrize(
    ("layers", "capacity", "digest", "makespan_ns", "candidate", "actions"),
    (
        (
            1,
            224,
            "bb1c105fac1dbe146b417bfc5dc862b0f35e9324d35480bef9d79b5708699e61",
            56_000,
            "tight-stall/packed-fit",
            13,
        ),
        (
            2,
            224,
            "69553a599141e41c6b655ba23fa941bedd61f44a4bfc4ffc3dc0cd5d17f52af9",
            110_000,
            "tight-stall/packed-fit",
            24,
        ),
        (
            5,
            800,
            "2714536a200d65afb2840223fb03563ec93c912338e8b19cffb8481708a8e362",
            152_000,
            "headroom-stall/packed-fifo",
            32,
        ),
        (
            10,
            500,
            "b2428807ae3801a4de45d312329683ad85a17ffa92b256c90c2aa5652baf005c",
            326_000,
            "headroom-stall/packed-fifo",
            94,
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
    )

    assert result.schedule.digest == digest
    assert result.simulation.makespan_ns == makespan_ns
    assert result.diagnostics.selected_candidate_id == candidate
    assert len(result.schedule.actions) == actions
