"""A corpus is whole documents, each ending in the end-of-text id, cut at
the requested token count, and readable by the harnesses."""

import numpy
import pytest

from baselines.common.data import Corpus
from baselines.scripts.prepare_corpus import prepare

pyarrow = pytest.importorskip("pyarrow")
parquet = pytest.importorskip("pyarrow.parquet")


class Tokenizer:
    eos_token_id = 99

    def __len__(self):
        return 100

    def __call__(self, texts, add_special_tokens=False):
        assert not add_special_tokens
        return {"input_ids": [[ord(c) % 90 for c in text] for text in texts]}


def write_shards(directory, documents_per_shard):
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, documents in enumerate(documents_per_shard):
        path = directory / f"shard-{index}.parquet"
        parquet.write_table(pyarrow.table({"text": documents}), path)
        paths.append(path)
    return paths


def test_the_corpus_is_whole_documents_cut_at_the_token_count(tmp_path):
    paths = write_shards(
        tmp_path / "shards", [["abcd", "efgh", "ij"], ["klmnop", "qr", "stuv"]]
    )
    meta = prepare(
        Tokenizer(),
        paths,
        tmp_path / "corpus",
        name="stub",
        tokens=12,
        val_tokens=3,
        log=lambda line: None,
    )
    val = numpy.fromfile(tmp_path / "corpus" / "val.bin", dtype=numpy.uint32)
    train = numpy.fromfile(tmp_path / "corpus" / "train.bin", dtype=numpy.uint32)
    # The first batch, the whole first shard, is the validation split.
    assert len(val) == 13 and (val == 99).sum() == 3
    # The second shard's documents go to train while they fit twelve tokens:
    # klmnop (7) and qr (3) fit, stuv (5) would not.
    assert train.tolist() == [
        *(ord(c) % 90 for c in "klmnop"),
        99,
        *(ord(c) % 90 for c in "qr"),
        99,
    ]
    assert meta["train_tokens"] == 10 and meta["val_tokens"] == 13
    assert meta["tokens_requested"] == 12
    assert meta["files"] == [str(path) for path in paths]
    assert meta["tokenizer"] == "stub" and meta["vocab_size"] == 100
    corpus = Corpus(tmp_path / "corpus")
    assert corpus.vocab_size == 100 and len(corpus) == 10
    corpus.check_vocabulary(100)


def test_requests_that_cannot_be_met_are_refused(tmp_path):
    with pytest.raises(ValueError):
        prepare(
            Tokenizer(), [], tmp_path / "corpus", name="stub", tokens=1, val_tokens=0
        )
    paths = write_shards(tmp_path / "shards", [["abc"]])
    with pytest.raises(ValueError):
        prepare(
            Tokenizer(), paths, tmp_path / "corpus", name="stub", tokens=0, val_tokens=0
        )
