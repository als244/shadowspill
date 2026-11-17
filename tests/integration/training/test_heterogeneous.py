"""Heterogeneous ranks share a step in units of the sequence cap."""

from __future__ import annotations

import pytest
import torch

from shadowspill_workloads.inputs.text.documents import IGNORE
from shadowspill_workloads.inputs.text.packed import PackedTokens
from shadowspill_workloads.inputs.text.units import held_out
from shadowspill_workloads.recipes.text.heterogeneous.data import (
    UnitUpdates,
    unit_ranges,
)
from tests.integration.training._synthetic import write_dataset

MAX_SEQ_LEN = 512


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
    held = held_out(
        data,
        name="n",
        max_seq_len=MAX_SEQ_LEN,
        capacity=MAX_SEQ_LEN,
        tokens=16 * MAX_SEQ_LEN,
        shares=[3, 1],
        rank=1,
    )
    # a quarter of sixteen units at one unit per microbatch
    assert len(held) == 4 and held[0][1].item() == pytest.approx(
        1 / data.validation_packer(MAX_SEQ_LEN, MAX_SEQ_LEN).trained_tokens(range(16))
    )


def test_the_aggregate_adds_the_ranks_evaluation_shares():
    from shadowspill_workloads.recipes.text.experiment.logging import sum_shares

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


def test_held_out_units_divide_among_ranks_and_pad_the_last_microbatch(data):
    units = data.validation_packer(MAX_SEQ_LEN, MAX_SEQ_LEN)

    def held(tokens, shares, rank):
        return held_out(
            data,
            name="n",
            max_seq_len=MAX_SEQ_LEN,
            capacity=2 * MAX_SEQ_LEN,
            tokens=tokens,
            shares=shares,
            rank=rank,
        )

    # Six units over two equal ranks, two units a microbatch: each rank's
    # second microbatch is one held-out unit and one padding unit.
    both = [held(6 * MAX_SEQ_LEN, [1, 1], rank) for rank in (0, 1)]
    assert [len(calls) for calls in both] == [2, 2]
    (tokens, targets, lengths), weight = both[1][1]
    assert torch.equal(tokens[:, :MAX_SEQ_LEN], units.microbatch(5)[0])
    assert bool((targets[:, MAX_SEQ_LEN:] == IGNORE).all())
    assert int(lengths[lengths.numel() // 2]) == MAX_SEQ_LEN
    assert weight.item() == pytest.approx(1 / units.trained_tokens(range(6)))
    trained = sum(
        int((targets != IGNORE).sum()) for calls in both for (_, targets, _), _ in calls
    )
    assert trained == units.trained_tokens(range(6))
    with pytest.raises(ValueError, match="whole sequences of max_seq_len"):
        held(6 * MAX_SEQ_LEN + 1, [1, 1], 0)
    with pytest.raises(ValueError, match="every rank's share"):
        held(5 * MAX_SEQ_LEN, [1, 1], 0)
