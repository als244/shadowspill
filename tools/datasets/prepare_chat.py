"""Render a chat dataset through the tokenizer's chat template, tokenize it,
and mark the assistant's tokens as the ones to learn.

    python -m tools.datasets.prepare_chat \
        --tokenizer <name or path> \
        --dataset <hub dataset> --files <file pattern> --out <directory> \
        [--revision REV] [--shards N] [--messages-column messages] \
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
verbatim, or rewrites earlier turns, is refused."""

import argparse
import json
from pathlib import Path

from tools.datasets.chat import tokenize_chat
from tools.datasets.corpus import write_meta


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
