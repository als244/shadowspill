"""Tokenize a dataset for a config, with its tokenizer, into its data directory.

    python -m training.prepare finetune.json --dataset <hub dataset> \\
        --files <pattern> [--shards N] [--revision REV] \\
        [--text-column text | --messages-column messages] [--val-tokens N] \\
        [--max-train-tokens N] [--tokenizer DIR] [--separator TOKEN]
    python -m training.prepare finetune.json --local a.parquet b.parquet [...]

The tokenizer is the config's checkpoint directory unless ``--tokenizer``
names another, and the output is the directory the config's ``data`` names.
``--messages-column`` selects the chat preparation, which renders each
conversation through the tokenizer's chat template and marks the assistant's
tokens; ``--local`` reads parquet files on disk instead of a Hub dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from workloads.recipes.text.data.prepare import tokenize, write_meta
from workloads.recipes.text.data.prepare_chat import tokenize_chat
from workloads.recipes.text.run import load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config")
    parser.add_argument("--dataset", help="a Hub dataset; or --local for files on disk")
    parser.add_argument("--files", help="file pattern in the dataset, {} for the shard")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--revision")
    parser.add_argument("--local", nargs="+", help="parquet files on disk")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--messages-column", help="conversations: the chat preparation")
    parser.add_argument("--val-tokens", type=int, default=1_000_000)
    parser.add_argument("--max-train-tokens", type=int)
    parser.add_argument("--tokenizer", help="defaults to the config's checkpoint")
    parser.add_argument("--separator", help="the token that ends each chat document")
    args = parser.parse_args(argv)
    options = load_config(args.config, ["run_dir=prepare"])
    out = Path(options["data"]["directory"])
    source = args.tokenizer or options.get("checkpoint", {}).get("directory")
    if source is None:
        source = options.get("model", {}).get("directory")
    if source is None:
        parser.error("the config names no checkpoint; pass --tokenizer")
    if bool(args.local) == bool(args.dataset):
        parser.error("give --dataset with --files, or --local files")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(source)
    if args.local:
        paths = [Path(p) for p in args.local]
    else:
        from huggingface_hub import hf_hub_download

        if not args.files:
            parser.error("--dataset needs --files")
        paths = [
            hf_hub_download(
                args.dataset,
                args.files.format(shard),
                repo_type="dataset",
                revision=args.revision,
            )
            for shard in range(args.shards)
        ]
    extra = {"dataset": args.dataset, "revision": args.revision}
    if args.messages_column:
        counts = tokenize_chat(
            tokenizer,
            paths,
            out,
            messages_column=args.messages_column,
            val_tokens=args.val_tokens,
            max_train_tokens=args.max_train_tokens,
            separator=args.separator,
        )
        extra.update(
            rendered="chat_template",
            mask="assistant",
            eos_id=counts["eos_id"],
            val_marked_tokens=counts["val_marked_tokens"],
            train_marked_tokens=counts["train_marked_tokens"],
        )
    else:
        counts = tokenize(
            tokenizer,
            paths,
            out,
            text_column=args.text_column,
            val_tokens=args.val_tokens,
            max_train_tokens=args.max_train_tokens,
        )
    meta = write_meta(out, tokenizer, str(source), counts, **extra)
    print(json.dumps(meta, indent=2))
    print(f"done in {counts['seconds']:.0f}s")


if __name__ == "__main__":
    main()
