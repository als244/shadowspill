"""Download a text dataset and tokenize it into the files a run packs.

    python -m workloads.recipes.text.prepare_data --tokenizer <name or path> \\
        --dataset <hub dataset> --files <file pattern> --out <directory> \\
        [--shards N] [--text-column text] [--val-tokens N]

Reads ``--shards`` parquet files of the dataset -- ``--files`` names them, with
``{}`` for the shard's index, formatted as Python does -- tokenizes the text
column, and writes ``train.bin`` and ``val.bin`` in ``--out``: every document's
token ids followed by the tokenizer's end-of-text id, back to back, as uint32.
The first ``--val-tokens`` tokens go to ``val.bin``. ``meta.json`` records the
tokenizer, the end-of-text id, the vocabulary size and what was read. Downloads
go wherever the Hugging Face cache is (``HF_HOME``).

``tokenize`` and ``write_meta`` are the pieces the baselines' corpus script
shares: it reads local parquet files and stops at a token count.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Iterable
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--files", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--val-tokens", type=int, default=5_000_000)
    return parser.parse_args()


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
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main() -> None:
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    args = arguments()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    files = [args.files.format(shard) for shard in range(args.shards)]
    downloads = (
        hf_hub_download(args.dataset, name, repo_type="dataset") for name in files
    )
    counts = tokenize(
        tokenizer,
        downloads,
        args.out,
        text_column=args.text_column,
        val_tokens=args.val_tokens,
    )
    meta = write_meta(args.out, tokenizer, args.tokenizer, counts, dataset=args.dataset)
    print(json.dumps(meta, indent=2))
    print(f"done in {counts['seconds']:.0f}s")


if __name__ == "__main__":
    main()
