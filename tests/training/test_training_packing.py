"""Whole documents packed into microbatches, from a small tokenized dataset."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tests.training._synthetic import EOS, LENGTHS, write_dataset
from training.data import PackedTokens
from training.documents import IGNORE

MAX_SEQ_LEN = 512
TOKENS = 2048


@pytest.fixture
def data(tmp_path: Path) -> PackedTokens:
    return PackedTokens(write_dataset(tmp_path / "tokens"))


def test_documents_longer_than_max_seq_len_are_dropped(data: PackedTokens) -> None:
    packer = data.train_packer(TOKENS, MAX_SEQ_LEN)
    kept = [length for length in LENGTHS if length <= MAX_SEQ_LEN]
    assert sorted(set(packer.source.lengths.tolist())) == sorted(set(kept))
    # the dropped document keeps its place in the ordinals of those after it
    assert packer.source.ordinals[3] == 4


def test_a_microbatch_holds_whole_documents_and_their_targets(
    data: PackedTokens,
) -> None:
    packer = data.train_packer(TOKENS, MAX_SEQ_LEN)
    for index in range(20):
        tokens, targets, seq_lens = packer.microbatch(index)
        documents = packer.documents(index)
        assert tokens.shape == targets.shape == (1, TOKENS)
        assert seq_lens.shape == (TOKENS // data.min_tokens_per_seq,)
        lengths = [length for _, length in documents]
        assert seq_lens[: len(lengths)].tolist() == lengths
        assert int(seq_lens.sum()) == TOKENS  # the padded tail is sequences too
        assert int(seq_lens.max()) <= MAX_SEQ_LEN
        at = 0
        for length in lengths:
            document = tokens[0, at : at + length]
            assert int(document[-1]) == EOS
            assert torch.equal(targets[0, at : at + length - 1], document[1:])
            assert int(targets[0, at + length - 1]) == IGNORE
            at += length
        assert (targets[0, at:] == IGNORE).all()
        assert packer.trained_tokens([index]) == int((targets != IGNORE).sum())


def test_packing_is_the_same_every_time(data: PackedTokens) -> None:
    first, second = (
        data.train_packer(TOKENS, MAX_SEQ_LEN),
        data.train_packer(TOKENS, MAX_SEQ_LEN),
    )
    assert [first.documents(i) for i in range(30)] == [
        second.documents(i) for i in range(30)
    ]


def test_stats_describe_what_microbatches_hold(data: PackedTokens) -> None:
    packer = data.train_packer(TOKENS, MAX_SEQ_LEN)
    stats = packer.stats(range(4))
    counts = [len(packer.documents(i)) for i in range(4)]
    lengths = [length for i in range(4) for _, length in packer.documents(i)]
    assert stats["sequences"] == sum(counts)
    assert stats["sequences_per_microbatch_min"] == min(counts)
    assert stats["sequences_per_microbatch_max"] == max(counts)
    assert stats["fill"] == sum(lengths) / (4 * TOKENS)
    assert stats["trained_tokens"] == packer.trained_tokens(range(4))


def test_planning_examples_are_whole_documents_of_max_seq_len(
    data: PackedTokens,
) -> None:
    examples = data.examples(3, 2, MAX_SEQ_LEN)
    assert len(examples) == 2
    tokens, targets, seq_lens = examples[0]
    assert tokens.shape == targets.shape == (1, 3 * MAX_SEQ_LEN)
    assert seq_lens.tolist()[:4] == [MAX_SEQ_LEN] * 3 + [0]
    assert examples[0][0] is not examples[1][0]


def test_a_long_document_can_be_truncated_instead(tmp_path: Path) -> None:
    data = PackedTokens(write_dataset(tmp_path / "tokens"), long_documents="truncate")
    source = data.train_packer(TOKENS, MAX_SEQ_LEN).source
    assert len(source.lengths) == len(LENGTHS) * 40  # nothing dropped
    long = LENGTHS.index(700)
    assert source.lengths[long] == MAX_SEQ_LEN and source.continues[long]
    targets = source.targets(long)
    start = int(source.starts[long])
    # every kept position is trained on the token that follows it in the document
    assert (targets == source.stream[start + 1 : start + MAX_SEQ_LEN + 1]).all()
    assert source.trained_tokens(long) == MAX_SEQ_LEN
    whole = LENGTHS.index(300)
    assert source.trained_tokens(whole) == 299 and not source.continues[whole]
    with pytest.raises(ValueError, match="long_documents"):
        PackedTokens(tmp_path / "tokens", long_documents="split").train_packer(
            TOKENS, MAX_SEQ_LEN
        )


def test_a_long_document_can_be_spliced_into_pieces(tmp_path: Path) -> None:
    data = PackedTokens(write_dataset(tmp_path / "tokens"), long_documents="splice")
    source = data.train_packer(TOKENS, MAX_SEQ_LEN).source
    assert len(source.lengths) == len(LENGTHS) * 40 + 40  # the 700 is two pieces
    first = LENGTHS.index(700)
    second = first + 1
    assert (source.ordinals[first], source.ordinals[second]) == (3, 3)
    assert (source.offsets[first], source.offsets[second]) == (0, MAX_SEQ_LEN)
    assert (source.lengths[first], source.lengths[second]) == (MAX_SEQ_LEN, 188)
    start = int(source.starts[first])
    # the first piece trains its last position on the second piece's first token
    assert (source.targets(first) == source.stream[start + 1 : start + 513]).all()
    assert source.trained_tokens(first) == MAX_SEQ_LEN
    assert source.trained_tokens(second) == 187  # the document's end
    assert int(source.targets(second)[-1]) == IGNORE
    assert source.record(first) == [3, MAX_SEQ_LEN]
    assert source.record(second) == [3, 188, MAX_SEQ_LEN]
