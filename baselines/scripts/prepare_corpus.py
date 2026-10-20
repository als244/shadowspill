"""Tokenize a local parquet text sample once for one tokenizer family.

    python -m baselines.scripts.prepare_corpus --tokenizer PATH_OR_ID \\
        --shards 'GLOB' --output DIR [--tokens N] [--val-tokens N] \\
        [--text-column text] [--name NAME]

Reads the parquet files the glob matches, in sorted order, and writes
``train.bin``, ``val.bin`` and ``meta.json`` in ``--output`` as
``workloads.recipes.text.prepare_data`` does: every document's token ids
followed by the tokenizer's end-of-text id, as uint32, the first
``--val-tokens`` to ``val.bin`` and ``train.bin`` stopping at the last whole
document within ``--tokens``. A harness reads the result with ``--data DIR``;
every model family that shares the tokenizer reads the same corpus, and
``meta.json`` names the tokenizer (``--name``, or the path given) so a row
can say what it trained on.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

from workloads.recipes.text.prepare_data import tokenize, write_meta


def prepare(
    tokenizer: Any,
    paths: list[str | Path],
    output: Path,
    *,
    name: str,
    tokens: int,
    val_tokens: int,
    text_column: str = "text",
    log=print,
) -> dict[str, Any]:
    """The corpus directory for ``tokenizer`` from ``paths``; ``meta.json``
    is returned."""

    if tokens < 1 or val_tokens < 0:
        raise ValueError("the token count must be positive and val tokens nonnegative")
    if not paths:
        raise ValueError("no parquet files to read")
    counts = tokenize(
        tokenizer,
        paths,
        output,
        text_column=text_column,
        val_tokens=val_tokens,
        max_train_tokens=tokens,
        log=log,
    )
    return write_meta(output, tokenizer, name, counts, tokens_requested=tokens)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tokenizer", required=True, help="a path or a Hub id")
    parser.add_argument("--shards", required=True, help="glob of parquet files")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokens", type=int, default=1_000_000_000)
    parser.add_argument("--val-tokens", type=int, default=5_000_000)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--name", help="what meta.json calls the tokenizer")
    args = parser.parse_args(argv)
    paths = sorted(glob.glob(args.shards, recursive=True))
    if not paths:
        parser.error(f"no parquet files match {args.shards}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    meta = prepare(
        tokenizer,
        paths,
        args.output,
        name=args.name or args.tokenizer,
        tokens=args.tokens,
        val_tokens=args.val_tokens,
        text_column=args.text_column,
        log=lambda line: print(line, flush=True),
    )
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
