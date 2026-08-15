from __future__ import annotations

import importlib

import pytest

from shadowspill.planner import PressureFitOptions, pressurefit
from shadowspill.planner._capi import planner_library_path
from shadowspill.planner.model import InitialPlacement

from ._examples import (
    training_chain_config,
    training_chain_initial,
    training_chain_program,
)

pytestmark = pytest.mark.skipif(
    planner_library_path() is None,
    reason="compiled planner library is not installed",
)


@pytest.mark.parametrize(
    ("layers", "capacity"),
    ((1, 224), (2, 224), (5, 800), (10, 500)),
)
@pytest.mark.parametrize(
    "placement",
    (InitialPlacement.REQUIRED, InitialPlacement.GREEDY),
)
def test_native_portfolio_matches_python_authority(
    monkeypatch: pytest.MonkeyPatch,
    layers: int,
    capacity: int,
    placement: InitialPlacement,
) -> None:
    program = training_chain_program(layers)
    initial = training_chain_initial(layers)
    config = training_chain_config(capacity)
    options = PressureFitOptions(initial_placement=placement, workers=1)

    native = pressurefit(
        program,
        initial_residency=initial,
        config=config,
        options=options,
    )

    implementation = importlib.import_module("shadowspill.planner.pressurefit")
    monkeypatch.setattr(implementation, "planner_library_path", lambda: None)
    reference = pressurefit(
        program,
        initial_residency=initial,
        config=config,
        options=options,
    )

    assert native.schedule == reference.schedule
    assert native.selections == reference.selections
    assert native.simulation == reference.simulation
    assert native.diagnostics.selected_candidate_id == (
        reference.diagnostics.selected_candidate_id
    )
    assert native.diagnostics.selected_selection_id == (
        reference.diagnostics.selected_selection_id
    )
    assert native.diagnostics.selected_makespan_ns == (
        reference.diagnostics.selected_makespan_ns
    )
    assert tuple(item.candidate_id for item in native.diagnostics.candidates) == tuple(
        item.candidate_id for item in reference.diagnostics.candidates
    )
