"""Candidate composition retains exact probe geometry and independent profiles."""

from dataclasses import replace

import pytest

from engine_tests.offline.test_sharded_forward import forward
from shadowspill import (
    Parallelism,
)
from shadowspill.orchestration.communication import communication_probes


def reordered(source):
    shard = source.bindings["weight_shards"]["weight_storage"]
    return replace(
        source,
        bindings={
            **source.bindings,
            "weight_shards": {
                "weight_storage": {**shard, "groups": ((3, 1, 0, 2),)},
            },
        },
    )


def test_union_preserves_rank_order_and_deduplicates_across_candidates():
    source = forward()
    first, requests = communication_probes(iter((source, source, reordered(source))))
    assert first is source
    assert requests == [
        {
            "operation": "all_gather",
            "ranks": ranks,
            "tensor_bytes": 64,
            "dtype": "torch.uint8",
            "inplace": True,
        }
        for ranks in ((0, 1, 2, 3), (3, 1, 0, 2))
    ]


@pytest.mark.parametrize(
    "change,message",
    [
        (
            {"parallelism": Parallelism.dp(ranks_per_replica=2, min_shard_bytes=0)},
            "same parallelism",
        ),
        ({"setup": {"gpu": "different"}}, "same build setup"),
    ],
)
def test_mixed_calibration_contexts_are_rejected(change, message):
    source = forward()
    with pytest.raises(ValueError, match=message):
        communication_probes((source, replace(source, **change)))


def test_empty_calibration_is_explicit():
    with pytest.raises(ValueError, match="at least one Program"):
        communication_probes(())
