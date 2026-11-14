"""Preparation records preserve requested choices before server resolution."""

from dataclasses import replace

import pytest

from shadowspill_client import (
    BuildCandidates,
    BuildCase,
    BuildOptions,
    PlanningOptions,
    Policy,
    Preparation,
    ProfilingOptions,
)
from shadowspill_client.protocol.codec import decode, encode
from shadowspill_client.protocol.definitions import DefinitionSource
from shadowspill_client.protocol.preparation import ResolvedPreparation


def test_preparation_roundtrip_keeps_explicit_unchunked_choices_and_counts():
    value = Preparation(
        build=BuildOptions(partition=Policy("external/checkpoint_blocks@1")),
        candidates=BuildCandidates(
            microbatch_sizes=[1, 2, 4],
            chunking=[None, Policy("external/chunks@1", {"size": 32})],
        ),
        planning=PlanningOptions(microbatch_counts=[1, 2, 4]),
        profiling=ProfilingOptions(),
    )
    restored = decode(encode(value))
    assert restored == value
    assert restored.candidates.chunking[0] is None
    assert restored.planning.microbatch_counts == (1, 2, 4)


def test_fixed_unchunked_and_unspecified_options_remain_distinct():
    assert BuildOptions().to_dict() == {}
    assert BuildOptions(chunking=None).to_dict() == {"chunking": None}
    assert BuildOptions.from_dict({"chunking": None}) == BuildOptions(chunking=None)
    with pytest.raises(ValueError, match="both a fixed value"):
        Preparation(
            build=BuildOptions(chunking=None),
            candidates=BuildCandidates(chunking=[None]),
        )


def test_complete_cases_express_coupled_choices_without_cross_expansion():
    value = Preparation(
        candidates=BuildCandidates(
            cases=[
                BuildCase(1, BuildOptions(chunking=None)),
                BuildCase(
                    2, BuildOptions(chunking=Policy("external/chunks@1", {"size": 8}))
                ),
            ]
        )
    )
    assert decode(encode(value)) == value
    assert len(value.candidates.cases) == 2


@pytest.mark.parametrize(
    "factory",
    [
        lambda: BuildCandidates(microbatch_sizes=[]),
        lambda: BuildCandidates(microbatch_sizes=[1, 1]),
        lambda: BuildCandidates(chunking=[None, None]),
        lambda: BuildCandidates(cases=[BuildCase(1)], microbatch_sizes=[1]),
        lambda: BuildOptions(chunking=lambda: None),
        lambda: BuildOptions.from_dict({"misspelled": None}),
        lambda: PlanningOptions(microbatch_counts=[True]),
        lambda: PlanningOptions(microbatch_counts=[0]),
        lambda: ProfilingOptions(minimum_samples=4, maximum_samples=3),
        lambda: ProfilingOptions(measurement_seconds=float("nan")),
        lambda: Preparation(
            build=BuildOptions(chunking=None),
            candidates=BuildCandidates(
                cases=[BuildCase(1, BuildOptions(chunking=None))]
            ),
        ),
    ],
)
def test_invalid_or_conflicting_choices_fail_before_any_preparation(factory):
    with pytest.raises((ValueError, TypeError)):
        factory()


def test_resolved_preparation_is_portable_immutable_and_complete():
    value = ResolvedPreparation(
        "external/preset@1",
        DefinitionSource("external", "1", "a" * 64),
        [
            BuildCase(
                2,
                BuildOptions(
                    partition=Policy("external/whole@1"),
                    chunking=None,
                    graph_pair_policy=None,
                ),
            )
        ],
        ProfilingOptions(),
        PlanningOptions(
            microbatch_counts=[1, 2],
            search={"engine_record": "opaque to SDK"},
            workers=0,
        ),
        {"cases.0.microbatch_size": "explicit"},
    )
    assert decode(encode(value)) == value
    with pytest.raises(TypeError):
        value.origins["cases.0.microbatch_size"] = "preset"
    for change in (
        {"cases": [BuildCase(2)]},
        {"cases": [*value.cases, *value.cases]},
        {"origins": {"setting": "guessed"}},
        {"planning": PlanningOptions(microbatch_counts=[1], search={})},
    ):
        with pytest.raises(ValueError):
            replace(value, **change)


def test_omitted_workers_is_distinct_from_explicit_all_cpus():
    assert PlanningOptions().workers is None
    assert PlanningOptions(workers=0).workers == 0
