from __future__ import annotations

from dataclasses import replace

from reference.python.pressurefit import pressurefit as pressurefit_reference
from shadowspill.planner import GenericPlanningOptions, pressurefit

from ...._examples import (
    training_chain_config,
    training_chain_initial,
    training_chain_program,
)


def test_compiled_pressurefit_matches_readable_reference() -> None:
    program = training_chain_program(3)
    config = training_chain_config(256)
    initial = training_chain_initial(3)

    indexed = pressurefit(
        program,
        initial_residency=initial,
        config=config,
        generic=GenericPlanningOptions(minimum_object_bytes_evict_eligible=0),
    )
    reference = pressurefit_reference(
        program,
        initial_residency=initial,
        config=config,
    )

    assert indexed.schedule == reference.schedule
    assert indexed.selections == reference.selections
    assert indexed.simulation == reference.simulation


def test_latency_metadata_does_not_change_search_or_reference_result() -> None:
    program = training_chain_program(3)
    config = training_chain_config(256)
    initial = training_chain_initial(3)
    metadata = replace(
        config,
        devices=tuple(
            replace(device, fetch_latency_ns=(1 << 64) - 1, evict_latency_ns=300_000)
            for device in config.devices
        ),
    )
    options = GenericPlanningOptions(minimum_object_bytes_evict_eligible=0)
    baseline = pressurefit(
        program, initial_residency=initial, config=config, generic=options, workers=1
    )
    for result in (
        pressurefit(
            program,
            initial_residency=initial,
            config=metadata,
            generic=options,
            workers=1,
        ),
        pressurefit_reference(program, initial_residency=initial, config=metadata),
    ):
        assert result.schedule == baseline.schedule
        assert result.selections == baseline.selections
        assert result.simulation == baseline.simulation
        assert result.simulation_config == metadata
