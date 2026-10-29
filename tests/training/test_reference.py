"""The transformers reference reproduces a tiny run's loss from its records,
mask included, which also holds the catalog's Llama 3 to transformers."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from tests.workloads.test_checkpoints import huggingface_state, materialized, metadata
from workloads.models import catalog
from workloads.recipes.text.data.packed import PackedTokens

EOS = 1
LENGTHS = (20, 33, 17, 40, 25, 30)


def test_reference_matches_the_catalog_model_on_a_masked_corpus(tmp_path):
    pytest.importorskip("transformers")
    from training.reference import reference_loss

    family, name = "llama3", "llama3-tiny"
    source = materialized(name)
    config = source.config
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    save_file(
        huggingface_state(family, source, config), checkpoint / "model.safetensors"
    )
    (checkpoint / "config.json").write_text(json.dumps(metadata(family, config)))

    tokens_dir = tmp_path / "tokens"
    tokens_dir.mkdir()
    generator = np.random.default_rng(0)
    documents = [
        np.append(generator.integers(2, config.vocab_size, length - 1), EOS)
        for length in LENGTHS
    ]
    stream = np.concatenate(documents).astype(np.uint32)
    stream.tofile(tokens_dir / "train.bin")
    mask = generator.integers(0, 2, len(stream)).astype(np.uint8)
    mask[stream == EOS] = 0
    mask.tofile(tokens_dir / "train.mask.bin")
    (tokens_dir / "meta.json").write_text(
        json.dumps({"vocab_size": config.vocab_size, "eos_id": EOS})
    )
    max_seq_len = 64
    data = PackedTokens(tokens_dir, long_documents="truncate", min_tokens_per_seq=16)
    packer = data.train_packer(128, max_seq_len)

    model = catalog.build_on_meta(
        name, "pytorch", dtype="float32", device="cpu", max_seq_len=max_seq_len
    )
    with catalog.reader(family)(checkpoint, max_seq_len=max_seq_len) as reader:
        reader.load_into(model)
    total = 0.0
    with torch.no_grad():
        for sequence in range(len(packer.documents(0))):
            ordinal = packer.documents(0)[sequence][0]
            tokens = torch.from_numpy(packer.source.tokens(ordinal).astype("int64"))
            targets = torch.from_numpy(packer.source.targets(ordinal))
            total += float(
                model.loss(
                    tokens[None],
                    targets[None],
                    seq_lens=torch.tensor([len(tokens)], dtype=torch.int32),
                    reduction="sum",
                )
            )
    trained = packer.trained_tokens([0])
    recorded = total / trained

    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_text(
        json.dumps(
            {
                "data": {"directory": str(tokens_dir), "long_documents": "truncate"},
                "max_seq_len": max_seq_len,
                "model": {"directory": str(checkpoint)},
            }
        )
    )
    (run / "packing.jsonl").write_text(
        json.dumps(
            {
                "step": 1,
                "microbatches": [packer.documents(0)],
                "trained_tokens": trained,
            }
        )
        + "\n"
    )
    (run / "metrics.jsonl").write_text(
        json.dumps({"step": 1, "train/loss": recorded}) + "\n"
    )
    result = reference_loss(run, step=1, device="cpu", dtype="float32", tag="check")
    assert result["trained_tokens"] == trained == result["recorded_trained_tokens"]
    assert result["documents"] == len(packer.documents(0)) > 1
    assert abs(result["difference"]) < 1e-3, result
    assert json.loads((run / "reference-check.json").read_text()) == result
