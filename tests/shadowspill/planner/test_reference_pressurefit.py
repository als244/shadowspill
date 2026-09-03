from __future__ import annotations

from reference.python.pressurefit import pressurefit as pressurefit_reference
from shadowspill.planner import PressureFitOptions, pressurefit

from ._examples import (
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
        options=PressureFitOptions(minimum_object_bytes_evict_eligible=0),
    )
    reference = pressurefit_reference(
        program,
        initial_residency=initial,
        config=config,
    )

    assert indexed.schedule == reference.schedule
    assert indexed.selections == reference.selections
    assert indexed.simulation == reference.simulation
