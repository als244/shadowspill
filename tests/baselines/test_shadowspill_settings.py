"""What a ShadowSpill case can ask for and what the engine refuses."""

import argparse

import pytest

from baselines.common.launch import command, engine_module
from baselines.engines.shadowspill.precision import optimizer_arguments
from baselines.engines.shadowspill.settings import (
    Settings,
    add_arguments,
    from_namespace,
)

BASE = {
    "model": "llama3-tiny",
    "tokens_per_step": 1024,
    "tokens_per_microbatch": 128,
    "sequence_length": 64,
}


def test_flags_round_trip_every_setting():
    settings = Settings.from_dict(
        {
            **BASE,
            "execution_gib": 2,
            "spill_gib": 2.5,
            "artifact_store": "/tmp/store",
            "parameter_rounding": "stochastic",
            "opt_state_rounding": "stochastic",
            "fuse_narrow_gradient_accumulation": False,
        }
    )
    settings.validate(2)
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    assert from_namespace(parser.parse_args(settings.arguments())) == settings
    assert "--recompute" not in settings.arguments()
    assert engine_module("shadowspill").ENVIRONMENT == "shadowspill"
    assert (
        command("shadowspill", "py", 2, settings, "out")[-len(settings.arguments()) :]
        == settings.arguments()
    )
    configuration = settings.describe(2)["configuration"]
    assert configuration["execution_gib"] == 2
    assert configuration["recompute"] is None


@pytest.mark.parametrize("options", ["quarters", "eighths", "halves", "0,1/2,1"])
def test_the_resolution_options_are_the_planners_own(options):
    settings = Settings.from_dict({**BASE, "resolution_options": options})
    settings.validate(1)
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    assert from_namespace(parser.parse_args(settings.arguments())) == settings
    assert Settings.from_dict(BASE).resolution_options == "quarters"


@pytest.mark.parametrize("options", ["thirds", "0,3/2", ""])
def test_resolution_options_the_planner_refuses_are_refused(options):
    with pytest.raises(ValueError):
        Settings.from_dict({**BASE, "resolution_options": options}).validate(1)


def test_a_recompute_fraction_is_refused():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    given = from_namespace(
        parser.parse_args([*Settings.from_dict(BASE).arguments(), "--recompute", "0.5"])
    )
    with pytest.raises(ValueError, match="planner decides"):
        given.validate(1)


@pytest.mark.parametrize(
    "override",
    [
        {"execution_gib": 0},
        {"spill_gib": -1},
        {"execution_gib": 1, "external_headroom_gib": 1},
        {"host_headroom_gib": -0.5},
        {"allocator_cap_gib": 4.0},
    ],
)
def test_invalid_settings_are_refused(override):
    with pytest.raises(ValueError):
        Settings.from_dict({**BASE, **override}).validate(1)


def test_an_unnamed_rounding_is_rounding_to_nearest():
    unnamed = Settings.from_dict({**BASE, "opt_state_dtype": "bfloat16"}).precision()
    given = optimizer_arguments(unnamed)
    assert given["parameter_rounding"] == given["opt_state_rounding"] == "nearest"
    named = Settings.from_dict(
        {**BASE, "opt_state_dtype": "bfloat16", "opt_state_rounding": "stochastic"}
    ).precision()
    assert optimizer_arguments(named)["opt_state_rounding"] == "stochastic"


def test_artifact_location_and_sharing_options_round_trip():
    assert Settings.from_dict(BASE).artifact_store == "default"
    for store in ("default", "none", "/tmp/store"):
        settings = Settings.from_dict({**BASE, "artifact_store": store})
        settings.validate(1)
        parser = argparse.ArgumentParser()
        add_arguments(parser)
        assert from_namespace(parser.parse_args(settings.arguments())) == settings
    with pytest.raises(ValueError, match="artifact_store"):
        Settings.from_dict({**BASE, "artifact_store": ""}).validate(1)


def test_one_case_searches_every_split_of_the_update():
    unpinned = {k: v for k, v in BASE.items() if k != "tokens_per_microbatch"}
    settings = Settings.from_dict(unpinned)
    settings.validate(1)
    # 1,024 tokens of 64-token sequences: 16 sequences, split every way.
    assert settings.microbatch_candidates(1) == (1024, 512, 256, 128, 64)
    assert settings.geometry(1).tokens_per_microbatch == 1024
    bounded = Settings.from_dict(
        {**unpinned, "min_tokens_per_microbatch": 128, "max_tokens_per_microbatch": 512}
    )
    assert bounded.microbatch_candidates(1) == (512, 256, 128)
    assert bounded.microbatch_candidates(2) == (512, 256, 128)
    assert Settings.from_dict(BASE).microbatch_candidates(1) == (128,)
    for case in (
        settings,
        bounded,
        Settings.from_dict({**unpinned, "orderings": "depth-first"}),
    ):
        parser = argparse.ArgumentParser()
        add_arguments(parser)
        assert from_namespace(parser.parse_args(case.arguments())) == case


@pytest.mark.parametrize(
    "override",
    [
        {"orderings": "random"},
        {"min_tokens_per_microbatch": 2048},
        {"tokens_per_microbatch": 128, "max_tokens_per_microbatch": 512},
        {"min_tokens_per_microbatch": 0},
    ],
)
def test_a_search_with_nothing_to_search_is_refused(override):
    unpinned = {k: v for k, v in BASE.items() if k != "tokens_per_microbatch"}
    with pytest.raises(ValueError):
        Settings.from_dict({**unpinned, **override}).validate(1)
