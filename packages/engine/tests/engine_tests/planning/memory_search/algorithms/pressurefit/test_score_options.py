import pytest

from shadowspill.planning.memory_search import GenericPlanningOptions
from shadowspill.planning.memory_search.algorithms.pressurefit.candidates import (
    _problem_options,
)
from shadowspill.planning.memory_search.algorithms.pressurefit.options import (
    PressureFitOptions,
)


@pytest.mark.parametrize(
    ("field", "bound"),
    [("max_repair_attempts", 1 << 32), ("capacity_refinement_bytes", 1 << 64)],
)
def test_fixed_width_options_stop_below_what_the_planner_can_read(field, bound):
    largest = PressureFitOptions(**{field: bound - 1})
    compiled, _ = _problem_options(GenericPlanningOptions(), largest)
    assert getattr(compiled, field) == bound - 1
    for value in (bound, bound + 5):
        with pytest.raises(ValueError, match=field):
            PressureFitOptions(**{field: value})


def test_options_round_trip_and_default_portfolio():
    options = PressureFitOptions()
    assert PressureFitOptions.from_dict(options.to_dict()) == options
    assert len(options.residency_strategies) * len(options.fetch_rules) == 8


@pytest.mark.parametrize("field", ["cut_score", "cut_ranking", "evaluate_coalesced"])
def test_removed_controls_are_rejected_instead_of_changing_replay(field):
    with pytest.raises(TypeError):
        PressureFitOptions(**{field: "obsolete"})
    record = PressureFitOptions().to_dict()
    record[field] = "obsolete"
    with pytest.raises(ValueError):
        PressureFitOptions.from_dict(record)


@pytest.mark.parametrize(
    "field,value",
    [
        ("residency_strategies", ("relaxed-stall",)),
        ("fetch_rules", ("interval-entry",)),
    ],
)
def test_removed_policies_are_rejected(field, value):
    with pytest.raises(ValueError):
        PressureFitOptions(**{field: value})
