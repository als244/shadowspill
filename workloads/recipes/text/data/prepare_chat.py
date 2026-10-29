"""Render a chat dataset through the tokenizer's chat template, tokenize it,
and mark the assistant's tokens as the ones to learn.

    python -m workloads.recipes.text.data.prepare_chat --tokenizer <name or path> \\
        --dataset <hub dataset> --files <file pattern> --out <directory> \\
        [--revision REV] [--shards N] [--messages-column messages] \\
        [--val-tokens N] [--max-train-tokens N] [--separator TOKEN]

Every conversation -- a list of ``{"role", "content"}`` messages -- becomes
one document: its rendering by the tokenizer's chat template, tokenized,
followed by a separator, as ``prepare`` writes documents. The separator is a
token no conversation contains: the tokenizer's end-of-text id when the
template does not use it, otherwise ``<|end_of_text|>`` or ``<|endoftext|>``
(Llama 3 and Qwen end every turn with their end-of-text id, which would cut
a conversation into turns), or the one ``--separator`` names; ``meta.json``
records it as ``eos_id``. Beside
``train.bin`` and ``val.bin`` go ``train.mask.bin`` and ``val.mask.bin``, one
byte per token: 1 for a token of an assistant turn, from where its generation
prompt ends to where the turn's rendering ends (its end-of-turn marker
included), 0 for every other token and for the separator. A run that packs
the directory trains only the marked tokens (see ``documents``). An
assistant turn runs from where its generation prompt ends in the whole
rendering to the end of the template's end-of-turn tail after its content,
so a template that decorates only the final turn (Qwen's empty think block)
is read as it renders; one that does not render a message's content
verbatim, or rewrites earlier turns, is refused.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Iterable, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from workloads.recipes.text.data.prepare import write_meta

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
        start = common_prefix(text, prompt)
        content = str(message["content"])
        at = text.find(content, start)
        if at < 0 or tail in text[start:at]:
            raise ValueError(
                "the chat template does not render the assistant's content "
                "verbatim after its generation prompt; the assistant's tokens "
                "cannot be located"
            )
        end = at + len(content)
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


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--revision", help="Dataset commit or tag on the Hub")
    parser.add_argument("--files", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--messages-column", default="messages")
    parser.add_argument("--val-tokens", type=int, default=1_000_000)
    parser.add_argument("--max-train-tokens", type=int)
    parser.add_argument("--separator", help="the token that ends each document")
    return parser.parse_args()


def main() -> None:
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    args = arguments()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    files = [args.files.format(shard) for shard in range(args.shards)]
    downloads = (
        hf_hub_download(args.dataset, name, repo_type="dataset", revision=args.revision)
        for name in files
    )
    counts = tokenize_chat(
        tokenizer,
        downloads,
        args.out,
        messages_column=args.messages_column,
        val_tokens=args.val_tokens,
        max_train_tokens=args.max_train_tokens,
        separator=args.separator,
    )
    meta = write_meta(
        args.out,
        tokenizer,
        args.tokenizer,
        counts,
        dataset=args.dataset,
        revision=args.revision,
        rendered="chat_template",
        mask="assistant",
        eos_id=counts["eos_id"],
        val_marked_tokens=counts["val_marked_tokens"],
        train_marked_tokens=counts["train_marked_tokens"],
    )
    print(json.dumps(meta, indent=2))
    print(f"done in {counts['seconds']:.0f}s")


if __name__ == "__main__":
    main()
