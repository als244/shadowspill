"""Probe derivation uses padded transfer geometry, not tensor/task guesses."""

from dataclasses import replace

import pytest

from shadowspill import Program
from shadowspill.communication.requests import CollectiveRequest, collective_requests

from .test_sharded_forward import forward


def test_saved_program_derives_unique_gathers_with_logical_order(tmp_path):
    source = forward()
    shard = source.bindings["weight_shards"]["weight_storage"]
    source = replace(
        source,
        bindings={
            **source.bindings,
            "weight_shards": {
                "weight_storage": {**shard, "groups": ((3, 1, 0, 2),)},
                "second_weight": {**shard, "groups": ((3, 1, 0, 2),)},
            },
        },
    )
    source.save(tmp_path / "program.ssp")
    requests = collective_requests(Program.load(tmp_path / "program.ssp"), None)
    assert requests == (
        CollectiveRequest("all_gather", (3, 1, 0, 2), 64, "torch.uint8"),
    )
    assert collective_requests(source, []) == ()
    assert collective_requests(
        source,
        [
            {
                "operation": "reduce_scatter",
                "ranks": [1, 0],
                "tensor_bytes": 16,
                "dtype": "torch.float32",
            }
        ],
    ) == (CollectiveRequest("reduce_scatter", (1, 0), 16, "torch.float32"),)


@pytest.mark.parametrize(
    "changes",
    [
        {"operation": "send"},
        {"ranks": ()},
        {"ranks": (1, 1)},
        {"ranks": (True, 1)},
        {"tensor_bytes": 6},
        {"tensor_bytes": 0},
        {"tensor_bytes": 4.0},
        {"dtype": "float32"},
    ],
)
def test_probe_rejects_ambiguous_or_unaligned_requests(changes):
    with pytest.raises(ValueError):
        CollectiveRequest(
            **{
                "operation": "reduce_scatter",
                "ranks": (0, 1),
                "tensor_bytes": 16,
                "dtype": "torch.float32",
                **changes,
            }
        )


def test_training_without_lowered_communication_requires_explicit_probes():
    source = replace(forward(), bindings={"kind": "step"})
    with pytest.raises(ValueError, match="no supported distributed training"):
        collective_requests(source, None)
    assert collective_requests(source, []) == ()
