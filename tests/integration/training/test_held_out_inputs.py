"""Heterogeneous ranks share a step in units of the sequence cap."""

from __future__ import annotations

import pytest
import torch

from shadowspill_workloads.inputs.text.documents import IGNORE
from shadowspill_workloads.inputs.text.packed import PackedTokens
from shadowspill_workloads.inputs.text.units import held_out
from tests.integration.training._synthetic import write_dataset

MAX_SEQ_LEN = 512


@pytest.fixture
def data(tmp_path):
    return PackedTokens(write_dataset(tmp_path / "tokens"))


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
