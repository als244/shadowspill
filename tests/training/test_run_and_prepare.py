"""The two entry points a newcomer meets: ``training.run`` on a path or a
mapping with keyword overrides, and ``training.prepare`` driven by the config."""

from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tests.training._text_recipe import _config
from training import run


def test_run_takes_a_path_or_a_mapping_with_keyword_overrides(tmp_path):
    path = _config(tmp_path)
    result = run(path, run_dir=str(tmp_path / "a"), steps=2)
    assert result.step == 2
    mapping = json.loads(path.read_text())
    mapping["data"] = mapping["data"]["directory"]  # a directory alone
    del mapping["objective"]
    result = run(mapping, run_dir=str(tmp_path / "b"), steps=1, log_every=1)
    assert result.step == 1
    recorded = json.loads((tmp_path / "b" / "config.json").read_text())
    assert recorded["data"]["@call"].endswith("PackedTokens")
    assert recorded["objective"].endswith("next_token_prediction")
    with pytest.raises(ValueError, match="unknown config keys"):
        run(path, run_dir=str(tmp_path / "c"), stps=2)


@pytest.fixture
def tokenizer_dir(tmp_path):
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    specials = ["<|eos|>", "<|user|>", "<|assistant|>", "<|end|>", "<unk>"]
    words = ["hello", "world", "fine", "thanks", "and", "you", "?"]
    vocab = {token: index for index, token in enumerate([*specials, *words])}
    core = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="<unk>"))
    core.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    core.add_special_tokens(specials[:4])
    fast = transformers.PreTrainedTokenizerFast(
        tokenizer_object=core, eos_token="<|eos|>", unk_token="<unk>"
    )
    fast.chat_template = (
        "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}<|end|>"
        "{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"
    )
    directory = tmp_path / "tokenizer"
    fast.save_pretrained(directory)
    return directory


def test_prepare_tokenizes_for_the_config(tmp_path, tokenizer_dir):
    from training.prepare import main

    data = tmp_path / "tokens"
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "checkpoint": {"family": "llama3", "directory": str(tokenizer_dir)},
                "data": str(data),
                "max_seq_len": 32,
                "max_tokens_per_step": 64,
                "steps": 1,
            }
        )
    )
    text = tmp_path / "text.parquet"
    pq.write_table(pa.table({"text": ["hello world", "fine thanks and you"] * 4}), text)
    main([str(config), "--local", str(text), "--val-tokens", "1"])
    meta = json.loads((data / "meta.json").read_text())
    assert meta["eos_id"] == 0 and meta["train_tokens"] + meta["val_tokens"] == 4 * (
        3 + 5
    )
    assert not (data / "train.mask.bin").exists()

    chat = tmp_path / "chat.parquet"
    conversations = [
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "fine thanks"},
        ]
    ] * 3
    pq.write_table(pa.table({"messages": conversations}), chat)
    main(
        [
            str(config),
            "--local",
            str(chat),
            "--messages-column",
            "messages",
            "--val-tokens",
            "1",
        ]
    )
    meta = json.loads((data / "meta.json").read_text())
    assert meta["mask"] == "assistant" and meta["eos_id"] == 0
    # one batch of three conversations, all in the validation split; per
    # conversation the assistant's "fine", "thanks" and "<|end|>" are marked
    mask = np.fromfile(data / "val.mask.bin", dtype=np.uint8)
    tokens = np.fromfile(data / "val.bin", dtype=np.uint32)
    assert len(mask) == len(tokens) == 3 * 8 and mask.sum() == 3 * 3
    assert meta["val_marked_tokens"] == 9 and meta["train_tokens"] == 0
