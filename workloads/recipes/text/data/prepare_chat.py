"""Render a chat dataset through the tokenizer's chat template, tokenize it,
and mark the assistant's tokens as the ones to learn.

    python -m workloads.recipes.text.data.prepare_chat --tokenizer <name or path> \\
        --dataset <hub dataset> --files <file pattern> --out <directory> \\
        [--revision REV] [--shards N] [--messages-column messages] \\
        [--val-tokens N] [--max-train-tokens N]

Every conversation -- a list of ``{"role", "content"}`` messages -- becomes
one document: its rendering by the tokenizer's chat template, tokenized,
followed by the end-of-text id, as ``prepare`` writes documents. Beside
``train.bin`` and ``val.bin`` go ``train.mask.bin`` and ``val.mask.bin``, one
byte per token: 1 for a token of an assistant turn, from where its generation
prompt ends to where the turn's rendering ends (its end-of-turn marker
included), 0 for every other token and for the separator. A run that packs
the directory trains only the marked tokens (see ``documents``). The template
must render a conversation's turns as prefixes of one another, as the common
ones do; one that rewrites earlier turns is refused.
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


def render(
    tokenizer: Any, messages: Sequence[Message]
) -> tuple[str, list[tuple[int, int]]]:
    """The conversation as the chat template renders it, and the character
    spans of the assistant's turns: each from where its generation prompt ends
    to where the turn's rendering ends."""

    text = tokenizer.apply_chat_template(messages, tokenize=False)
    spans = []
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prompt = tokenizer.apply_chat_template(
            messages[:index], tokenize=False, add_generation_prompt=True
        )
        through = tokenizer.apply_chat_template(messages[: index + 1], tokenize=False)
        if not (text.startswith(through) and through.startswith(prompt)):
            raise ValueError(
                "the chat template does not render a conversation's turns as "
                "prefixes of one another; the assistant's tokens cannot be located"
            )
        spans.append((len(prompt), len(through)))
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
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Render and tokenize the conversations of the parquet files at ``paths``,
    in order, into ``out/val.bin`` and ``out/train.bin`` with their masks.
    Whole batches go to the validation split until it holds ``val_tokens``;
    with ``max_train_tokens``, training stops at the last whole conversation
    that fits. Returns the files read and the token counts, marked ones too."""

    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("marking the assistant's tokens needs a fast tokenizer")
    eos = tokenizer.eos_token_id
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
                rendered = [render(tokenizer, messages) for messages in conversations]
                encoded = tokenizer(
                    [text for text, _ in rendered],
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                )
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
    return {
        "files": read,
        "val_tokens": counts["val"],
        "train_tokens": counts["train"],
        "val_marked_tokens": counts["val_marked"],
        "train_marked_tokens": counts["train_marked"],
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
        val_marked_tokens=counts["val_marked_tokens"],
        train_marked_tokens=counts["train_marked_tokens"],
    )
    print(json.dumps(meta, indent=2))
    print(f"done in {counts['seconds']:.0f}s")


if __name__ == "__main__":
    main()
