from __future__ import annotations

import pytest

from reference.python.pressurefit.selection import SelectionCandidate, select_compiled
from shadowspill._libraries import shadowspill_library_path
from shadowspill.planner import PressureFitOptions, pressurefit

from ._examples import (
    training_chain_config,
    training_chain_initial,
    training_chain_program,
)

pytestmark = pytest.mark.skipif(
    shadowspill_library_path() is None,
    reason="the library is not installed",
)


def test_compiled_selector_matches_python_policy_ordering() -> None:
    program = training_chain_program(2)
    initial = training_chain_initial(2)
    config = training_chain_config(224)
    faster = pressurefit(program, initial_residency=initial, config=config)
    slower = pressurefit(
        program,
        initial_residency=initial,
        config=config,
        options=PressureFitOptions(
            residency_strategies=("tight-stall",),
            prefetch_rules=("demand",),
            evaluate_coalesced=False,
        ),
    )

    selected = select_compiled(
        (
            SelectionCandidate(
                program,
                slower.schedule,
                slower.selections,
                config,
                40,
                50,
            ),
            SelectionCandidate(
                program,
                faster.schedule,
                faster.selections,
                config,
                41,
                51,
            ),
        )
    )

    assert selected.selected_index == 1
    assert selected.selected_candidate_id == 41
    assert selected.selected_selection_id == 51
    assert selected.selected_makespan_ns == faster.simulation.makespan_ns
    assert selected.valid_candidate_count == 2
    assert tuple(item[2] for item in selected.candidate_results) == (
        slower.simulation.makespan_ns,
        faster.simulation.makespan_ns,
    )
