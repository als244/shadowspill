"""Tokenize a text corpus and record its tokenizer/data identity."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from shadowspill_workloads.inputs.text.tokenizer import tokenizer_identity


def tokenize(
    tokenizer: Any,
    paths: Iterable[str | Path],
    out: Path,
    *,
    text_column: str = "text",
    val_tokens: int = 5_000_000,
    max_train_tokens: int | None = None,
    batch_size: int = 2048,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Tokenize the text column of the parquet files at ``paths``, in order,
    into ``out/val.bin`` and ``out/train.bin``: every document's ids followed
    by the tokenizer's end-of-text id, as uint32. Whole batches go to
    ``val.bin`` until it holds ``val_tokens``; with ``max_train_tokens``,
    ``train.bin`` stops at the last whole document that fits. Returns the
    files read and the token counts."""

    eos = tokenizer.eos_token_id
    out.mkdir(parents=True, exist_ok=True)
    counts = {"val": 0, "train": 0}
    read: list[str] = []
    started = time.time()
    done = False
    with (
        open(out / "val.bin", "wb") as val,
        open(out / "train.bin", "wb") as train,
    ):
        for path in paths:
            read.append(str(path))
            batches = pq.ParquetFile(path).iter_batches(
                batch_size=batch_size, columns=[text_column]
            )
            for index, batch in enumerate(batches):
                texts = batch.column(text_column).to_pylist()
                documents = tokenizer(texts, add_special_tokens=False)["input_ids"]
                for document in documents:
                    document.append(eos)
                split = "val" if counts["val"] < val_tokens else "train"
                if split == "train" and max_train_tokens is not None:
                    room = max_train_tokens - counts["train"]
                    kept = []
                    for document in documents:
                        if len(document) > room:
                            done = True
                            break
                        kept.append(document)
                        room -= len(document)
                    documents = kept
                tokens = np.fromiter(
                    chain.from_iterable(documents),
                    dtype=np.uint32,
                    count=sum(map(len, documents)),
                )
                (val if split == "val" else train).write(tokens.tobytes())
                counts[split] += len(tokens)
                if index % 50 == 0 or done:
                    total = counts["val"] + counts["train"]
                    rate = total / max(time.time() - started, 1e-9)
                    log(
                        f"{Path(path).name} batch {index}: {total / 1e6:,.0f}M "
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
        "seconds": time.time() - started,
    }


def write_meta(
    out: Path, tokenizer: Any, name: str, counts: dict[str, Any], **extra: Any
) -> dict[str, Any]:
    """``meta.json`` beside the token files: the tokenizer, its end-of-text
    id and vocabulary size, the dtype, what was read and the counts."""

    meta = {
        "tokenizer": name,
        "vocab_size": len(tokenizer),
        "eos_id": tokenizer.eos_token_id,
        "dtype": "uint32",
        "files": counts["files"],
        "val_tokens": counts["val_tokens"],
        "train_tokens": counts["train_tokens"],
        **extra,
    }
    directory = Path(name)
    if all(
        (directory / file).is_file()
        for file in ("tokenizer.json", "tokenizer_config.json")
    ):
        meta["tokenizer_sha256"] = tokenizer_identity(directory)
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta
