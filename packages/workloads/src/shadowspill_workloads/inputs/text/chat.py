"""Render chat conversations and preserve assistant token supervision."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

Message = dict[str, Any]

#: Document-end tokens a tokenizer may hold beside an end-of-turn end-of-text id.
DOCUMENT_ENDS = ("<|end_of_text|>", "<|endoftext|>")


def choose_separator(
    tokenizer: Any, documents: Sequence[Sequence[int]], separator: str | None = None
) -> int:
    """The id that ends each document in the token file: ``separator`` when
    named, else the end-of-text id unless a conversation contains it, else a
    document-end token the vocabulary holds and no conversation contains."""

    present = {int(token) for ids in documents for token in ids}
    if separator is not None:
        chosen = tokenizer.convert_tokens_to_ids(separator)
        if chosen is None or chosen == getattr(tokenizer, "unk_token_id", None):
            raise ValueError(f"the tokenizer has no token {separator!r}")
        if chosen in present:
            raise ValueError(f"{separator!r} occurs inside a conversation")
        return int(chosen)
    candidates = [
        tokenizer.eos_token_id,
        *(tokenizer.convert_tokens_to_ids(t) for t in DOCUMENT_ENDS),
    ]
    for candidate in candidates:
        if (
            candidate is not None
            and candidate != getattr(tokenizer, "unk_token_id", None)
            and candidate not in present
        ):
            return int(candidate)
    raise ValueError(
        "the end-of-text id ends the template's turns and the vocabulary has no "
        "free document-end token; pass --separator with a token no conversation "
        "contains"
    )


PROBE = "\u2063probe\u2063"


def turn_tail(tokenizer: Any) -> str:
    """What the template renders after an assistant message's content: its
    end-of-turn marker and whatever follows, learned from a one-turn probe."""

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "?"}, {"role": "assistant", "content": PROBE}],
        tokenize=False,
    )
    at = rendered.find(PROBE)
    if at < 0:
        raise ValueError(
            "the chat template does not render an assistant message's content"
        )
    return rendered[at + len(PROBE) :]


def common_prefix(first: str, second: str) -> int:
    """How many leading characters the two strings share."""

    length = min(len(first), len(second))
    for index in range(length):
        if first[index] != second[index]:
            return index
    return length


def render(
    tokenizer: Any, messages: Sequence[Message], tail: str | None = None
) -> tuple[str, list[tuple[int, int]]]:
    """The conversation as the chat template renders it, and the character
    spans of the assistant's turns: each from where its generation prompt
    ends in the whole rendering to the end of the template's end-of-turn
    tail after its content."""

    if tail is None:
        tail = turn_tail(tokenizer)
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    spans = []
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prompt = tokenizer.apply_chat_template(
            messages[:index], tokenize=False, add_generation_prompt=True
        )
        prefix = tokenizer.apply_chat_template(
            messages[:index], tokenize=False, add_generation_prompt=False
        )
        boundary = common_prefix(text, prefix)
        content = str(message["content"])
        # A template may trim the content's surrounding whitespace.
        found = sorted(
            (at, candidate)
            for candidate in (content, content.strip())
            if candidate and (at := text.find(candidate, boundary)) >= 0
        )
        if not found or tail in text[boundary : found[0][0]]:
            raise ValueError(
                "the chat template does not render the assistant's content "
                "verbatim after its generation prompt; the assistant's tokens "
                "cannot be located"
            )
        at, candidate = found[0]
        # A generation decoration may share a prefix with the actual content:
        # e.g. "<think>" versus "<!DOCTYPE". Never consume those content bytes.
        start = min(common_prefix(text, prompt), at)
        end = at + len(candidate)
        if not text.startswith(tail, end):
            raise ValueError(
                "the chat template does not end the assistant's turn as its "
                "probe did; the assistant's tokens cannot be located"
            )
        spans.append((start, end + len(tail)))
    return text, spans


def assistant_mask(
    offsets: Sequence[tuple[int, int]], spans: Sequence[tuple[int, int]]
) -> np.ndarray:
    """One byte per token: 1 when the token starts inside an assistant turn."""

    mask = np.zeros(len(offsets), dtype=np.uint8)
    starts = np.fromiter(
        (start for start, _ in offsets), dtype=np.int64, count=len(offsets)
    )
    for begin, end in spans:
        mask[(starts >= begin) & (starts < end)] = 1
    return mask


def tokenize_chat(
    tokenizer: Any,
    paths: Iterable[str | Path],
    out: Path,
    *,
    messages_column: str = "messages",
    val_tokens: int = 1_000_000,
    max_train_tokens: int | None = None,
    batch_size: int = 256,
    separator: str | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Render and tokenize the conversations of the parquet files at ``paths``,
    in order, into ``out/val.bin`` and ``out/train.bin`` with their masks.
    Whole batches go to the validation split until it holds ``val_tokens``;
    with ``max_train_tokens``, training stops at the last whole conversation
    that fits. Returns the files read, the token counts, marked ones too, and
    the separator id (``eos_id``) chosen on the first batch, which every
    document ends with."""

    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("marking the assistant's tokens needs a fast tokenizer")
    eos: int | None = None
    tail = turn_tail(tokenizer)
    out.mkdir(parents=True, exist_ok=True)
    counts = {"val": 0, "train": 0, "val_marked": 0, "train_marked": 0}
    read: list[str] = []
    started = time.time()
    done = False
    with ExitStack() as stack:
        streams = {
            (split, kind): stack.enter_context(open(out / f"{split}{kind}.bin", "wb"))
            for split in ("val", "train")
            for kind in ("", ".mask")
        }
        for path in paths:
            read.append(str(path))
            batches = pq.ParquetFile(path).iter_batches(
                batch_size=batch_size, columns=[messages_column]
            )
            for index, batch in enumerate(batches):
                conversations = batch.column(messages_column).to_pylist()
                rendered = [
                    render(tokenizer, messages, tail) for messages in conversations
                ]
                encoded = tokenizer(
                    [text for text, _ in rendered],
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                )
                if eos is None:
                    eos = choose_separator(tokenizer, encoded["input_ids"], separator)
                    token = tokenizer.convert_ids_to_tokens(eos)
                    log(f"documents end with {token!r} ({eos})")
                elif any(eos in ids for ids in encoded["input_ids"]):
                    raise ValueError(f"separator {eos} occurs inside a conversation")
                documents = [
                    (
                        np.array([*ids, eos], dtype=np.uint32),
                        np.append(assistant_mask(offsets, spans), 0).astype(np.uint8),
                    )
                    for ids, offsets, (_, spans) in zip(
                        encoded["input_ids"],
                        encoded["offset_mapping"],
                        rendered,
                        strict=True,
                    )
                ]
                split = "val" if counts["val"] < val_tokens else "train"
                if split == "train" and max_train_tokens is not None:
                    room = max_train_tokens - counts["train"]
                    kept = []
                    for document in documents:
                        if len(document[0]) > room:
                            done = True
                            break
                        kept.append(document)
                        room -= len(document[0])
                    documents = kept
                if documents:
                    ids = np.concatenate([tokens for tokens, _ in documents])
                    marks = np.concatenate([mask for _, mask in documents])
                    streams[split, ""].write(ids.tobytes())
                    streams[split, ".mask"].write(marks.tobytes())
                    counts[split] += len(ids)
                    counts[split + "_marked"] += int(marks.sum())
                if index % 20 == 0 or done:
                    total = counts["val"] + counts["train"]
                    rate = total / max(time.time() - started, 1e-9)
                    log(
                        f"{Path(path).name} batch {index}: {total / 1e6:,.1f}M "
                        f"tokens ({rate / 1e6:.2f}M/s)"
                    )
                if done:
                    break
            if done:
                break
    if eos is None:
        raise ValueError("no conversations were read")
    return {
        "files": read,
        "val_tokens": counts["val"],
        "train_tokens": counts["train"],
        "val_marked_tokens": counts["val_marked"],
        "train_marked_tokens": counts["train_marked"],
        "eos_id": eos,
        "seconds": time.time() - started,
    }
