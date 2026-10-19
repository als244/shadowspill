"""Fair global-step accounting for independent training baselines."""

import pytest

from baselines.common import accumulation_steps, recompute_layers, summarize_cycles


@pytest.mark.parametrize("tokens,expected", [(4096, 64), (16384, 16), (131072, 2)])
def test_global_step_stays_at_two_million_tokens(tokens, expected):
    assert accumulation_steps(2097152, 4096, tokens, 8) == expected


@pytest.mark.parametrize("tokens", [0, 4095, 12288, 524288])
def test_invalid_geometries_are_rejected(tokens):
    with pytest.raises(ValueError):
        accumulation_steps(2097152, 4096, tokens, 8)


def test_slowest_rank_is_selected_per_step_before_taking_median():
    report = summarize_cycles([[1, 10, 1], [10, 1, 10]], global_tokens=100)
    assert report["step_seconds"] == 10
    assert report["global_tokens_per_second"] == 10


def test_partial_rank_measurements_cannot_be_reported_as_a_completed_run():
    with pytest.raises(ValueError):
        summarize_cycles([[1, 2], [1]], global_tokens=100)


@pytest.mark.parametrize(
    "fraction,count", [(0, 0), (0.25, 12), (0.5, 24), (0.75, 36), (1, 48)]
)
def test_recompute_choices_cover_expected_layer_counts(fraction, count):
    assert len(recompute_layers(48, fraction)) == count
