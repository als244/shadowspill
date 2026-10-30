"""Heterogeneous ranks share a step in units of the sequence cap."""

from __future__ import annotations

import pytest
import torch

from shadowspill.training import config
from tests.training._synthetic import write_dataset
from workloads.recipes.text.data.documents import IGNORE
from workloads.recipes.text.data.packed import PackedTokens
from workloads.recipes.text.heterogeneous.data import (
    UnitUpdates,
    unit_ranges,
    validation,
)
from workloads.recipes.text.heterogeneous.run import check_section, node_config

MAX_SEQ_LEN = 512
NODES = [
    {"address": "10.0.0.1", "ranks": 1},
    {"address": "10.0.0.2", "ranks": 1, "ssh": "peer", "repo": "/r", "python": "/p"},
]


@pytest.fixture
def data(tmp_path):
    return PackedTokens(write_dataset(tmp_path / "tokens"))


def test_units_divide_by_shares_and_concatenate_into_each_ranks_geometry(data):
    assert unit_ranges([3, 1], 8) == [(0, 6), (6, 2)]
    with pytest.raises(ValueError, match="divide by the shares"):
        unit_ranges([3, 1], 6)
    fast = UnitUpdates(
        data,
        name="n",
        max_seq_len=MAX_SEQ_LEN,
        capacity=3 * MAX_SEQ_LEN,
        step_units=8,
        shares=[3, 1],
        rank=0,
    )
    slow = UnitUpdates(
        data,
        name="n",
        max_seq_len=MAX_SEQ_LEN,
        capacity=MAX_SEQ_LEN,
        step_units=8,
        shares=[3, 1],
        rank=1,
    )
    units = data.train_packer(MAX_SEQ_LEN, MAX_SEQ_LEN)
    for step in range(2):
        a, b = next(fast), next(slow)
        # the source yields the trainer's calls: each microbatch with the
        # step's shared weight, one over the step's trained tokens
        expected_weight = 1 / units.trained_tokens(range(step * 8, step * 8 + 8))
        for update in (a, b):
            assert len(update["args"]) == 2
            assert all(
                weight.item() == pytest.approx(expected_weight)
                for _, weight in update["args"]
            )
        (tokens, targets, lengths), _ = a["args"][0]
        assert tokens.shape == (1, 3 * MAX_SEQ_LEN) and lengths.shape == (
            3 * (MAX_SEQ_LEN // data.min_tokens_per_seq),
        )
        expected = torch.cat([units.microbatch(step * 8 + u)[0] for u in range(3)], 1)
        assert torch.equal(tokens, expected)
        (tokens, targets, lengths), _ = b["args"][1]
        assert tokens.shape == (1, MAX_SEQ_LEN)
        assert torch.equal(tokens, units.microbatch(step * 8 + 7)[0])
        assert int((targets != IGNORE).sum()) == units.trained_tokens([step * 8 + 7])
    assert fast.last_documents[0] and slow.trained_tokens == fast.trained_tokens
    with pytest.raises(ValueError, match="whole microbatches"):
        UnitUpdates(
            data,
            name="n",
            max_seq_len=MAX_SEQ_LEN,
            capacity=4 * MAX_SEQ_LEN,
            step_units=8,
            shares=[3, 1],
            rank=0,
        )
    held = validation(
        data,
        name="n",
        max_seq_len=MAX_SEQ_LEN,
        capacity=MAX_SEQ_LEN,
        step_units=8,
        shares=[3, 1],
        rank=1,
        steps=2,
    )
    # two units a step at one unit per microbatch, over two steps
    assert len(held) == 4 and held[0][1].item() == pytest.approx(
        1 / data.validation_packer(MAX_SEQ_LEN, MAX_SEQ_LEN).trained_tokens(range(16))
    )


def test_the_section_is_checked_and_each_node_gets_its_share():
    raw = {
        "heterogeneous": {"nodes": NODES, "shares": [3, 1]},
        "max_tokens_per_step": 8192,
        "max_seq_len": 1024,
    }
    nodes, shares, port = check_section(raw)
    assert shares == [3, 1] and port == 29400
    assert node_config(raw, nodes, shares, 0)["rank_tokens_per_step"] == 6144
    assert node_config(raw, nodes, shares, 1)["rank_tokens_per_step"] == 2048
    assert (
        node_config(raw, nodes, shares, 1)["ranks"] == 2
        and node_config(raw, nodes, shares, 1)["node"] == 1
    )
    # Whole weights and gradients on every rank, the optimizer state sharded,
    # unless the config names its own layout.
    storage = config.resolve(node_config(raw, nodes, shares, 0)["build"])
    layout = storage["parallelism"].default_parameter_storage
    assert storage["parallelism"].world_size == 2
    assert (layout.weights, layout.gradients, layout.optimizer) == (
        "replicated",
        "replicated",
        "sharded",
    )
    own = {"@call": "shadowspill:Parallelism.dp", "ranks_per_replica": 2}
    chosen = node_config({**raw, "build": {"parallelism": own}}, nodes, shares, 1)
    assert chosen["build"]["parallelism"] == own
    with pytest.raises(ValueError, match="one positive integer per rank"):
        check_section({**raw, "heterogeneous": {"nodes": NODES, "shares": [3]}})
    with pytest.raises(ValueError, match="whole units for every share"):
        check_section({**raw, "max_tokens_per_step": 5120})
    two = [{"address": "a", "ranks": 2}, NODES[1]]
    with pytest.raises(ValueError, match="equal shares"):
        check_section(
            {
                "heterogeneous": {"nodes": two, "shares": [2, 1, 1]},
                "max_tokens_per_step": 8192,
                "max_seq_len": 1024,
            }
        )


def test_nodes_walk_the_union_of_their_slots():
    from workloads.recipes.text.heterogeneous.communicate import (
        shared_slots,
        slots,
        stand_in,
    )

    fast = slots(
        {"eval_every": 3}, {"candidates": {"6144_tokens": {}, "3072_tokens": {}}}
    )
    slow = slots({"eval_every": 3}, {"candidates": {"2048_tokens": {}}})
    assert fast == [
        "6144_tokens/training",
        "6144_tokens/evaluation",
        "3072_tokens/training",
        "3072_tokens/evaluation",
    ]
    union = shared_slots([fast, slow])
    assert len(union) == 6 and union == sorted(union)
    assert stand_in("6144_tokens/evaluation", slow) == "2048_tokens/evaluation"
    assert stand_in("3072_tokens/training", slow) == "2048_tokens/training"
    with pytest.raises(ValueError, match="built no evaluation"):
        stand_in(
            "6144_tokens/evaluation", slots({}, {"candidates": {"2048_tokens": {}}})
        )


def test_the_aggregate_adds_the_ranks_evaluation_shares():
    from workloads.recipes.text.experiment.logging import sum_shares

    evaluation = sum_shares(
        [
            {"step": 3, "eval/loss": 1.5, "eval/seconds": 0.2},
            {"step": 3, "eval/loss": 0.5, "eval/seconds": 0.4},
        ]
    )
    assert evaluation == {"step": 3, "eval/loss": 2.0, "eval/seconds": 0.4}
    training = sum_shares(
        [{"step": 4, "train/loss": 1.0}, {"step": 4, "train/loss": 2.0}]
    )
    assert training == {"step": 4, "train/loss": 3.0}
