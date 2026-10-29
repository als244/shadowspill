"""Chat preparation marks the assistant's tokens, and a masked corpus trains
only those."""

from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from workloads.recipes.text.data.documents import IGNORE, TokenDocuments
from workloads.recipes.text.data.packed import PackedTokens
from workloads.recipes.text.data.prepare import write_meta
from workloads.recipes.text.data.prepare_chat import (
    assistant_mask,
    choose_separator,
    render,
    tokenize_chat,
)

TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}<|end|>"
    "{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"
)
WORDS = [
    "hello",
    "world",
    "how",
    "are",
    "you",
    "fine",
    "thanks",
    "and",
    "well",
    "thinks",
    "?",
]


@pytest.fixture
def tokenizer():
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    specials = [
        "<|eos|>",
        "<|user|>",
        "<|assistant|>",
        "<|end|>",
        "<unk>",
        "<|endoftext|>",
    ]
    vocab = {token: index for index, token in enumerate([*specials, *WORDS])}
    core = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="<unk>"))
    core.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    core.add_special_tokens([*specials[:4], specials[5]])
    fast = transformers.PreTrainedTokenizerFast(
        tokenizer_object=core, eos_token="<|eos|>", unk_token="<unk>"
    )
    fast.chat_template = TEMPLATE
    return fast


CONVERSATIONS = [
    [
        {"role": "user", "content": "hello world"},
        {"role": "assistant", "content": "fine thanks"},
    ],
    [
        {"role": "user", "content": "how are you"},
        {"role": "assistant", "content": "well"},
        {"role": "user", "content": "and you"},
        {"role": "assistant", "content": "fine"},
    ],
]


def test_the_assistants_turns_are_located_and_marked(tokenizer):
    text, spans = render(tokenizer, CONVERSATIONS[1])
    assert text == (
        "<|user|>how are you<|end|><|assistant|>well<|end|>"
        "<|user|>and you<|end|><|assistant|>fine<|end|>"
    )
    assert [text[begin:end] for begin, end in spans] == ["well<|end|>", "fine<|end|>"]
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    marked = assistant_mask(encoded["offset_mapping"], spans)
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
    assert [token for token, flag in zip(tokens, marked, strict=True) if flag] == [
        "well",
        "<|end|>",
        "fine",
        "<|end|>",
    ]


def test_a_template_that_trims_the_content_still_locates_the_turn(tokenizer):
    tokenizer.chat_template = TEMPLATE.replace(
        "{{ m['content'] }}", "{{ m['content'] | trim }}"
    )
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "  fine thanks \n"},
    ]
    text, spans = render(tokenizer, messages)
    assert [text[a:b] for a, b in spans] == ["fine thanks<|end|>"]


def test_a_template_that_rewrites_earlier_turns_is_refused(tokenizer):
    tokenizer.chat_template = (
        "{% for m in messages %}{% if loop.last %}<|{{ m['role'] }}|>"
        "{{ m['content'] }}<|end|>{% else %}<|{{ m['role'] }}|>...<|end|>"
        "{% endif %}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"
    )
    with pytest.raises(ValueError, match="cannot be located"):
        render(tokenizer, CONVERSATIONS[1])


def test_a_template_decorating_only_the_final_turn_is_read_as_it_renders(tokenizer):
    """Qwen's templates give the final assistant turn an empty think block the
    earlier turns lack, and Qwen 3.5 puts it in the generation prompt while
    Qwen 3 leaves it to the model: the span starts where the generation
    prompt ends in the whole rendering either way."""
    base = (
        "{% for m in messages %}<|{{ m['role'] }}|>{% if m['role'] == 'assistant' "
        "and loop.last %}thinks {% endif %}{{ m['content'] }}<|end|>{% endfor %}"
    )
    tokenizer.chat_template = (
        base + "{% if add_generation_prompt %}<|assistant|>{% endif %}"
    )
    text, spans = render(tokenizer, CONVERSATIONS[1])
    assert [text[a:b] for a, b in spans] == ["well<|end|>", "thinks fine<|end|>"]
    tokenizer.chat_template = (
        base + "{% if add_generation_prompt %}<|assistant|>thinks {% endif %}"
    )
    text, spans = render(tokenizer, CONVERSATIONS[1])
    assert [text[a:b] for a, b in spans] == ["well<|end|>", "fine<|end|>"]


def test_prepared_files_carry_a_mask_a_run_trains_by(tokenizer, tmp_path):
    source = tmp_path / "chat.parquet"
    pq.write_table(pa.table({"messages": CONVERSATIONS * 3}), source)
    out = tmp_path / "tokens"
    counts = tokenize_chat(tokenizer, [source], out, val_tokens=1, batch_size=4)
    assert counts["eos_id"] == tokenizer.eos_token_id == 0
    write_meta(
        out, tokenizer, "tiny", counts, mask="assistant", eos_id=counts["eos_id"]
    )
    for split in ("val", "train"):
        tokens = np.fromfile(out / f"{split}.bin", dtype=np.uint32)
        mask = np.fromfile(out / f"{split}.mask.bin", dtype=np.uint8)
        assert len(tokens) == len(mask) > 0
        assert mask[tokens == tokenizer.eos_token_id].sum() == 0
    # three of each conversation: 8 and 15 tokens plus a separator each
    assert counts["val_tokens"] + counts["train_tokens"] == 3 * (9 + 16)
    assert counts["train_marked_tokens"] + counts["val_marked_tokens"] == 3 * (3 + 4)
    meta = json.loads((out / "meta.json").read_text())
    assert meta["mask"] == "assistant" and meta["eos_id"] == 0

    documents = TokenDocuments(
        out / "train.bin", 0, 32, "drop", mask=out / "train.mask.bin"
    )
    first = documents.targets(0)
    # <|user|> hello world <|end|> <|assistant|> fine thanks <|end|> <|eos|>
    words = tokenizer.convert_ids_to_tokens([int(t) for t in first if t != IGNORE])
    assert words == ["fine", "thanks", "<|end|>"]
    assert documents.trained_tokens(0) == 3
    assert first.tolist()[4:7] == [int(t) for t in documents.tokens(0)[5:8]]
    assert first.tolist()[:4] == [IGNORE] * 4 and first.tolist()[7:] == [IGNORE] * 2

    data = PackedTokens(out, min_tokens_per_seq=8)
    packer = data.train_packer(64, 32)
    tokens, targets, _ = packer.microbatch(0)
    assert packer.trained_tokens([0]) == int((targets != IGNORE).sum()) > 0
    unmasked = TokenDocuments(out / "train.bin", 0, 32, "drop")
    assert (
        unmasked.trained_tokens(0) == 8 and (unmasked.targets(0) != IGNORE).sum() == 8
    )


def test_a_template_ending_turns_with_the_end_of_text_id_gets_another_separator(
    tokenizer, tmp_path
):
    """Llama 3 and Qwen end every turn with their end-of-text id; splitting a
    file at it would make each turn a document, so a free document-end token
    separates conversations instead, and the corpus records it as eos_id."""
    tokenizer.chat_template = TEMPLATE.replace("<|end|>", "<|eos|>")
    text, _ = render(tokenizer, CONVERSATIONS[0])
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    assert tokenizer.eos_token_id in ids
    assert choose_separator(tokenizer, [ids]) == tokenizer.convert_tokens_to_ids(
        "<|endoftext|>"
    )
    with pytest.raises(ValueError, match="occurs inside"):
        choose_separator(tokenizer, [ids], "<|eos|>")
    with pytest.raises(ValueError, match="no token"):
        choose_separator(tokenizer, [ids], "<|nothing|>")
    source = tmp_path / "chat.parquet"
    pq.write_table(pa.table({"messages": CONVERSATIONS}), source)
    out = tmp_path / "tokens"
    counts = tokenize_chat(tokenizer, [source], out, val_tokens=0, batch_size=4)
    assert counts["eos_id"] == tokenizer.convert_tokens_to_ids("<|endoftext|>")
    documents = TokenDocuments(out / "train.bin", counts["eos_id"], 64, "drop")
    assert len(documents) == 2  # one document per conversation, turns intact
    assert documents.trained_tokens(0) == 8
