"""Download a text dataset and tokenize it into the files a run packs.

    python -m tools.datasets.prepare_text \
        --tokenizer <name or path> \
        --dataset <hub dataset> --files <file pattern> --out <directory> \
        [--revision REV] [--shards N] [--text-column text] [--val-tokens N] \
        [--max-train-tokens N]

Reads ``--shards`` parquet files of the dataset -- ``--files`` names them, with
``{}`` for the shard's index, formatted as Python does -- tokenizes the text
column, and writes ``train.bin`` and ``val.bin`` in ``--out``: every document's
token ids followed by the tokenizer's end-of-text id, back to back, as uint32.
The first ``--val-tokens`` tokens go to ``val.bin``; ``--max-train-tokens``
stops ``train.bin`` at the last whole document that fits. ``meta.json`` records the
tokenizer, the end-of-text id, the vocabulary size and what was read. Downloads
go wherever the Hugging Face cache is (``HF_HOME``).

``tokenize`` and ``write_meta`` are the pieces the baselines' corpus script
shares: it reads local parquet files and stops at a token count."""

import argparse
import json
from pathlib import Path

from tools.datasets.corpus import tokenize, write_meta


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--revision", help="Dataset commit or tag on the Hub")
    parser.add_argument("--files", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--val-tokens", type=int, default=5_000_000)
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
    counts = tokenize(
        tokenizer,
        downloads,
        args.out,
        text_column=args.text_column,
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
    )
    print(json.dumps(meta, indent=2))
    print(f"done in {counts['seconds']:.0f}s")


if __name__ == "__main__":
    main()
