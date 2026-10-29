"""Whole documents packed into microbatches, from a small tokenized dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tests.training._synthetic import EOS, LENGTHS, write_dataset
from workloads.recipes.text.data.documents import IGNORE, TokenDocuments
from workloads.recipes.text.data.packed import PackedTokens

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


def test_a_mask_keeps_only_the_marked_tokens_as_targets(tmp_path: Path) -> None:
    """Instruction tuning: a byte per token beside the token file says which
    tokens are the model's to produce; the others' targets leave the loss and
    the trained-token count, and the separator is never a target."""
    directory = write_dataset(tmp_path / "tokens")
    tokens = np.fromfile(directory / "train.bin", dtype=np.uint32)
    mask = np.random.default_rng(1).integers(0, 2, len(tokens)).astype(np.uint8)
    mask[tokens == EOS] = 0
    mask.tofile(directory / "train.mask.bin")
    packer = PackedTokens(directory).train_packer(TOKENS, MAX_SEQ_LEN)
    plain = PackedTokens(directory).validation_packer(TOKENS, MAX_SEQ_LEN)
    assert packer.source.mask is not None and plain.source.mask is None
    for index in range(10):
        batch_tokens, targets, _ = packer.microbatch(index)
        kept = targets[0] != IGNORE
        # a kept target is the next token, kept exactly where the mask marks it
        assert torch.equal(targets[0][kept], torch.roll(batch_tokens[0], -1)[kept])
        at = 0
        for ordinal, length in packer.documents(index):
            sequence = int(np.flatnonzero(packer.source.ordinals == ordinal)[0])
            start = int(packer.source.starts[sequence])
            expected = torch.from_numpy(mask[start + 1 : start + length].astype(bool))
            assert torch.equal(kept[at : at + length - 1], expected)
            assert not kept[at + length - 1]
            at += length
        assert not kept[at:].any()
        assert 0 < packer.trained_tokens([index]) == int(kept.sum()) < at
    with pytest.raises(ValueError, match="one byte per token"):
        TokenDocuments(
            directory / "train.bin", EOS, MAX_SEQ_LEN, mask=directory / "val.bin"
        )


def test_ranks_share_a_steps_microbatches_and_its_normalizer(
    data: PackedTokens,
) -> None:
    """Ranks take consecutive shares of every step's microbatches, equal or
    not, with the step's trained-token count as every rank's normalizer;
    their validation batches are distinct and normalized over all ranks'."""
    from workloads.recipes.text.data.source import PackedUpdates, validation_update

    geometry = dict(name="n", tokens=TOKENS, max_seq_len=MAX_SEQ_LEN)
    whole = PackedUpdates(data, counts=[4], **geometry)
    uneven = [
        PackedUpdates(data, counts=[3, 1], rank=rank, **geometry) for rank in (0, 1)
    ]
    for _ in range(3):
        step = next(whole)
        parts = [next(half) for half in uneven]
        assert [p["normalizer"] for p in parts] == [step["normalizer"]] * 2
        joined = parts[0]["candidates"]["n"] + parts[1]["candidates"]["n"]
        assert [len(p["candidates"]["n"]) for p in parts] == [3, 1]
        for mine, theirs in zip(joined, step["candidates"]["n"], strict=True):
            assert all(torch.equal(a, b) for a, b in zip(mine, theirs, strict=True))
        assert uneven[0].trained_tokens == whole.trained_tokens
    assert uneven[0].last_documents != uneven[1].last_documents
    state = uneven[1].state_dict()
    assert state["counts"] == [3, 1] and state["rank"] == 1
    with pytest.raises(ValueError, match="another rank"):
        uneven[0].load_state_dict(state)
    with pytest.raises(ValueError, match="different microbatch geometry"):
        whole.load_state_dict(state)
    with pytest.raises(ValueError, match="at least one microbatch"):
        PackedUpdates(data, counts=[2, 0], **geometry)
    held = [
        validation_update(data, microbatches=2, rank=rank, world_size=2, **geometry)
        for rank in (0, 1)
    ]
    alone = validation_update(data, microbatches=4, **geometry)
    assert held[0]["normalizer"] == held[1]["normalizer"] == alone["normalizer"]
    first = [h["candidates"]["n"][0][0] for h in held]
    assert not torch.equal(first[0], first[1])
    assert torch.equal(first[1], alone["candidates"]["n"][2][0])


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


@pytest.mark.parametrize("minimum", [512, 1024])
def test_packing_rejects_geometry_without_document_slots(tmp_path, minimum):
    data = PackedTokens(write_dataset(tmp_path / "tokens"), min_tokens_per_seq=minimum)
    with pytest.raises(ValueError, match="reduce min_tokens_per_seq"):
        data.train_packer(512, 512)
