"""The shape of one optimizer update, and the tier each round starts from."""

import argparse

import pytest

from baselines.common import geometry
from baselines.common.geometry import Geometry, accumulation_steps, recompute_layers


@pytest.mark.parametrize("tokens,expected", [(4096, 64), (16384, 16), (131072, 2)])
def test_global_step_stays_at_two_million_tokens(tokens, expected):
    assert accumulation_steps(2097152, 4096, tokens, 8) == expected


@pytest.mark.parametrize("tokens", [0, 4095, 12288, 524288])
def test_invalid_geometries_are_rejected(tokens):
    with pytest.raises(ValueError):
        accumulation_steps(2097152, 4096, tokens, 8)


@pytest.mark.parametrize(
    "fraction,count", [(0, 0), (0.25, 12), (0.5, 24), (0.75, 36), (1, 48)]
)
def test_recompute_choices_cover_expected_layer_counts(fraction, count):
    assert len(recompute_layers(48, fraction)) == count


def test_recompute_layers_are_spread_through_the_model():
    assert recompute_layers(8, 0.5) == (1, 3, 5, 7)
    assert recompute_layers(4, 1.0) == (0, 1, 2, 3)
    with pytest.raises(ValueError):
        recompute_layers(4, 1.5)


def test_geometry_derives_the_rank_and_microbatch_counts():
    shape = Geometry(4096, 131072, 8192)
    assert shape.accumulation == 16
    assert shape.sequences_per_microbatch == 2
    assert shape.tokens_per_rank_per_step == 131072
    assert shape.as_dict() == {
        "sequence_length": 4096,
        "tokens_per_step": 131072,
        "tokens_per_microbatch": 8192,
        "world_size": 1,
        "accumulation": 16,
        "sequences_per_microbatch": 2,
        "tokens_per_rank_per_step": 131072,
    }
    spread = Geometry(4096, 2097152, 16384, world_size=8)
    assert spread.accumulation == 16
    assert spread.tokens_per_rank_per_step == 262144


@pytest.mark.parametrize(
    "fields",
    [
        (4096, 131072, 12288),
        (4096, 131072, 8192, 3),
        (4096.0, 131072, 8192),
        (4096, 131072, True),
    ],
)
def test_a_geometry_that_does_not_divide_or_is_not_integral_is_refused(fields):
    with pytest.raises(ValueError):
        Geometry(*fields)


def test_tiers_fix_the_tokens_per_update():
    assert Geometry.tier("rtx5090", tokens_per_microbatch=8192).accumulation == 16
    assert Geometry.tier("h100", tokens_per_microbatch=16384).tokens_per_step == 262144
    node = Geometry.tier("h100x8", tokens_per_microbatch=16384, world_size=8)
    assert node.accumulation == 16
    with pytest.raises(ValueError):
        Geometry.tier("tpu", tokens_per_microbatch=8192)


def parse(*argv):
    parser = argparse.ArgumentParser()
    geometry.add_arguments(parser)
    return parser.parse_args(argv)


def test_flags_take_the_tier_or_an_explicit_step_total():
    from_tier = geometry.from_arguments(
        parse("--tokens-per-microbatch", "8192", "--tier", "rtx5090"), 1
    )
    assert from_tier == Geometry(4096, 131072, 8192)
    explicit = geometry.from_arguments(
        parse(
            "--tokens-per-microbatch",
            "8192",
            "--tokens-per-step",
            "65536",
            "--sequence-length",
            "2048",
        ),
        2,
    )
    assert explicit == Geometry(2048, 65536, 8192, 2)
    agreeing = parse(
        "--tokens-per-microbatch",
        "8192",
        "--tier",
        "h100",
        "--tokens-per-step",
        "262144",
    )
    assert geometry.from_arguments(agreeing, 1).tokens_per_step == 262144


def test_flags_need_a_step_total_and_may_not_contradict_the_tier():
    with pytest.raises(ValueError):
        geometry.from_arguments(parse("--tokens-per-microbatch", "8192"), 1)
    contradiction = parse(
        "--tokens-per-microbatch", "8192", "--tier", "h100", "--tokens-per-step", "1024"
    )
    with pytest.raises(ValueError):
        geometry.from_arguments(contradiction, 1)
