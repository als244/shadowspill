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


def test_pressurefit_fails_closed_without_compiled_planner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    implementation = importlib.import_module("shadowspill.planner.pressurefit")

    def missing_library() -> None:
        raise RuntimeError("compiled planner unavailable")

    monkeypatch.setattr(implementation, "load_planner_library", missing_library)
    with pytest.raises(RuntimeError, match="compiled planner unavailable"):
        pressurefit(
            training_chain_program(1),
            initial_residency=training_chain_initial(1),
            config=training_chain_config(224),
        )


@pytest.mark.parametrize(
    ("layers", "capacity"),
    ((1, 224), (2, 224), (5, 800), (10, 500)),
)
@pytest.mark.parametrize(
    "placement",
    (InitialPlacement.REQUIRED, InitialPlacement.GREEDY),
)
def test_native_portfolio_is_deterministic(
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

    repeated = pressurefit(
        program,
        initial_residency=initial,
        config=config,
        options=options,
    )

    assert native.schedule == repeated.schedule
    assert native.selections == repeated.selections
    assert native.simulation == repeated.simulation
    assert native.diagnostics.selected_candidate_id == (
        repeated.diagnostics.selected_candidate_id
    )
    assert native.diagnostics.selected_selection_id == (
        repeated.diagnostics.selected_selection_id
    )
    assert native.diagnostics.selected_makespan_ns == (
        repeated.diagnostics.selected_makespan_ns
    )
    assert tuple(
        candidate.candidate_id
        for context in native.diagnostics.recomputation_contexts
        for candidate in context.candidate_evaluations
    ) == tuple(
        candidate.candidate_id
        for context in repeated.diagnostics.recomputation_contexts
        for candidate in context.candidate_evaluations
    )
