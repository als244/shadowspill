from __future__ import annotations

from reference.python.pressurefit import pressurefit as pressurefit_reference
from shadowspill.planner import pressurefit

from ._examples import (
    training_chain_config,
    training_chain_initial,
    training_chain_program,
)


def test_compiled_pressurefit_matches_readable_reference() -> None:
    program = training_chain_program(3)
    config = training_chain_config(256)
    initial = training_chain_initial(3)

    compiled = pressurefit(program, initial_residency=initial, config=config)
    reference = pressurefit_reference(
        program,
        initial_residency=initial,
        config=config,
    )

    assert compiled.schedule == reference.schedule
    assert compiled.selections == reference.selections
    assert compiled.simulation == reference.simulation
