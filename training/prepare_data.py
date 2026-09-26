"""Download a text dataset and tokenize it into the files a run packs.

    python -m training.prepare_data --tokenizer <name or path> \\
        --dataset <hub dataset> --files <file pattern> --out <directory> \\
        [--shards N] [--text-column text] [--val-tokens N]

Reads ``--shards`` parquet files of the dataset -- ``--files`` names them, with
``{}`` for the shard's index, formatted as Python does -- tokenizes the text
column, and writes ``train.bin`` and ``val.bin`` in ``--out``: every document's
token ids followed by the tokenizer's end-of-text id, back to back, as uint32.
The first ``--val-tokens`` tokens go to ``val.bin``. ``meta.json`` records the
tokenizer, the end-of-text id, the vocabulary size and what was read. Downloads
go wherever the Hugging Face cache is (``HF_HOME``).
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import chain
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer


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


def main() -> None:
    args = arguments()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    eos = tokenizer.eos_token_id
    args.out.mkdir(parents=True, exist_ok=True)
    counts = {"val": 0, "train": 0}
    started = time.time()
    files = [args.files.format(shard) for shard in range(args.shards)]
    with (
        open(args.out / "val.bin", "wb") as val,
        open(args.out / "train.bin", "wb") as train,
    ):
        for shard, name in enumerate(files):
            path = hf_hub_download(args.dataset, name, repo_type="dataset")
            batches = pq.ParquetFile(path).iter_batches(
                batch_size=2048, columns=[args.text_column]
            )
            for index, batch in enumerate(batches):
                texts = batch.column(args.text_column).to_pylist()
                documents = tokenizer(texts, add_special_tokens=False)["input_ids"]
                for document in documents:
                    document.append(eos)
                tokens = np.fromiter(
                    chain.from_iterable(documents),
                    dtype=np.uint32,
                    count=sum(map(len, documents)),
                )
                split = "val" if counts["val"] < args.val_tokens else "train"
                (val if split == "val" else train).write(tokens.tobytes())
                counts[split] += len(tokens)
                if index % 50 == 0:
                    total = counts["val"] + counts["train"]
                    rate = total / (time.time() - started)
                    print(
                        f"shard {shard} batch {index}: {total / 1e6:,.0f}M tokens "
                        f"({rate / 1e6:.2f}M/s)",
                        flush=True,
                    )
    meta = {
        "tokenizer": args.tokenizer,
        "vocab_size": len(tokenizer),
        "eos_id": eos,
        "dataset": args.dataset,
        "files": files,
        "dtype": "uint32",
        "val_tokens": counts["val"],
        "train_tokens": counts["train"],
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    print(f"done in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
