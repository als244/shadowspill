from __future__ import annotations

import pytest

from shadowspill.planner import PressureFitOptions, pressurefit
from shadowspill.pytorch.profiling import TaskMeasurement
from shadowspill.pytorch.spatial_admission import replay_selected_schedule
from tests.planner._examples import (
    config,
    exact_capacity_program,
    exact_capacity_residency,
)


def test_selected_schedule_replays_object_generations_and_outputs() -> None:
    initial, final = exact_capacity_residency()
    selected = pressurefit(
        exact_capacity_program(),
        initial_residency=initial,
        final_residency=final,
        config=config(),
        options=PressureFitOptions(
            residency_strategies=("relaxed-stall",),
            prefetch_rules=("latest-safe",),
            evaluate_coalesced=False,
        ),
    )
    replay = replay_selected_schedule(
        selected,
        {
            "task_abi": TaskMeasurement(
                1_000,
                0,
                0,
                (),
                (1_000,),
                "unit-test",
            )
        },
        slab_bytes=122,
        alignment=1,
    )
    assert replay.peak_allocated_bytes == 122
    assert replay.final_allocated_bytes == 61


def test_selected_schedule_requires_every_profile_measurement() -> None:
    initial, final = exact_capacity_residency()
    selected = pressurefit(
        exact_capacity_program(),
        initial_residency=initial,
        final_residency=final,
        config=config(),
    )
    with pytest.raises(ValueError, match="lacks task measurement"):
        replay_selected_schedule(selected, {}, slab_bytes=122, alignment=1)
