import pytest

from shadowspill.planner.search.algorithms.pressurefit.options import PressureFitOptions


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
