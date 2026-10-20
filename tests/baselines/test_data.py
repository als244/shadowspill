"""Synthetic tokens per rank and windows of a pre-tokenized corpus."""

import json

import numpy
import pytest
import torch

from baselines.common.data import Corpus, synthetic_tokens


def test_synthetic_tokens_are_deterministic_per_seed_and_distinct_per_rank():
    tokens, targets = synthetic_tokens(512, 64, seed=1, rank=0)
    assert tokens.shape == targets.shape == (1, 64)
    assert tokens.dtype == torch.int64
    assert int(tokens.max()) < 512 and int(targets.max()) < 512
    again, _ = synthetic_tokens(512, 64, seed=1, rank=0)
    assert torch.equal(tokens, again)
    other_rank, _ = synthetic_tokens(512, 64, seed=1, rank=1)
    other_seed, _ = synthetic_tokens(512, 64, seed=2, rank=0)
    assert not torch.equal(tokens, other_rank)
    assert not torch.equal(tokens, other_seed)
    with pytest.raises(ValueError):
        synthetic_tokens(0, 64, seed=1, rank=0)
    with pytest.raises(ValueError):
        synthetic_tokens(512, 0, seed=1, rank=0)


def write_corpus(directory, tokens, **meta):
    directory.mkdir(parents=True, exist_ok=True)
    numpy.asarray(list(tokens), dtype=numpy.uint32).tofile(directory / "train.bin")
    (directory / "meta.json").write_text(
        json.dumps({"tokenizer": "test", "vocab_size": 100, "dtype": "uint32", **meta})
    )
    return directory


def test_windows_are_disjoint_across_ranks_and_steps_until_the_data_wraps(tmp_path):
    corpus = Corpus(write_corpus(tmp_path / "corpus", range(17)))
    assert len(corpus) == 17
    assert corpus.vocab_size == 100
    tokens, targets, wrapped = corpus.window(rank=0, world_size=2, step=0, tokens=4)
    assert tokens.tolist() == [[0, 1, 2, 3]]
    assert targets.tolist() == [[1, 2, 3, 4]]
    assert tokens.dtype == torch.int64
    assert wrapped == 0
    tokens, _, wrapped = corpus.window(rank=1, world_size=2, step=0, tokens=4)
    assert tokens.tolist() == [[4, 5, 6, 7]]
    tokens, targets, wrapped = corpus.window(rank=1, world_size=2, step=1, tokens=4)
    assert tokens.tolist() == [[12, 13, 14, 15]]
    assert targets.tolist() == [[13, 14, 15, 16]]
    assert wrapped == 0
    tokens, _, wrapped = corpus.window(rank=0, world_size=2, step=2, tokens=4)
    assert tokens.tolist() == [[0, 1, 2, 3]]
    assert wrapped == 1


def test_corpus_refuses_what_a_model_cannot_read(tmp_path):
    corpus = Corpus(write_corpus(tmp_path / "corpus", range(17)))
    corpus.check_vocabulary(100)
    corpus.check_vocabulary(1000)
    with pytest.raises(ValueError):
        corpus.check_vocabulary(99)
    with pytest.raises(ValueError):
        corpus.window(rank=0, world_size=1, step=0, tokens=32)
    for request in (
        {"rank": 1, "world_size": 1, "step": 0, "tokens": 4},
        {"rank": 0, "world_size": 0, "step": 0, "tokens": 4},
        {"rank": 0, "world_size": 1, "step": -1, "tokens": 4},
        {"rank": 0, "world_size": 1, "step": 0, "tokens": 0},
    ):
        with pytest.raises(ValueError):
            corpus.window(**request)


def test_corpus_directories_must_be_complete(tmp_path):
    with pytest.raises(FileNotFoundError):
        Corpus(tmp_path / "missing")
    with pytest.raises(ValueError):
        Corpus(write_corpus(tmp_path / "narrow", range(17), dtype="uint16"))
    with pytest.raises(ValueError):
        Corpus(write_corpus(tmp_path / "short", [1]))
    incomplete = write_corpus(tmp_path / "incomplete", range(17))
    (incomplete / "meta.json").write_text(
        json.dumps({"vocab_size": 100, "dtype": "uint32"})
    )
    with pytest.raises(ValueError):
        Corpus(incomplete)
