"""Retokenize an existing FineWeb-Edu sample for each official Qwen tokenizer.

Head node only: tokenizer downloads and CPU work, no CUDA. Source documents are
recovered from the existing GPT-2 stream at its EOT boundaries. This is a small
validation dataset, not a new pretraining corpus or a checkpoint download.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path.home() / "storage/datasets/fineweb_edu_gpt2_chicago_sample")
    parser.add_argument("--outdir", type=Path, default=Path.home() / "storage/shadowspill/qwen_moe_ep8_1007/data")
    parser.add_argument("--tokens", type=int, default=50_000_000)
    args = parser.parse_args()
    evidence = Path(__file__).resolve().parents[1] / "sources"
    source_meta = json.loads((args.source / "meta.json").read_text())
    original = AutoTokenizer.from_pretrained(source_meta["tokenizer"])
    source = np.memmap(args.source / "train.bin", dtype=np.uint32, mode="r")
    # Locate only the prefix needed for the short validation experiment.
    prefix = source[:min(len(source), args.tokens * 3)]
    ends = np.flatnonzero(prefix == source_meta["eos_id"])
    for model in ("qwen3moe", "qwen35moe"):
        root = args.outdir / model
        if (root / "meta.json").exists():
            print(f"SKIP complete {root}", flush=True)
            continue
        root.mkdir(parents=True, exist_ok=True)
        provenance = json.loads((evidence / f"{model}-source.json").read_text())
        tokenizer = AutoTokenizer.from_pretrained(provenance["repository"], revision=provenance["revision"])
        tokenizer.save_pretrained(root / "tokenizer")
        written, offset, documents = 0, 0, 0
        destination = root / "train.bin.partial"
        digest = hashlib.sha256()
        with destination.open("wb") as stream:
            for first in range(0, len(ends), 256):
                batch = []
                for end in ends[first:first + 256]:
                    ids = prefix[offset:int(end)]
                    offset = int(end) + 1
                    if len(ids):
                        batch.append(original.decode(ids.tolist(), skip_special_tokens=False))
                if not batch:
                    continue
                encoded = tokenizer(batch, add_special_tokens=False, truncation=False)["input_ids"]
                for ids in encoded:
                    values = np.asarray([*ids, tokenizer.eos_token_id], dtype=np.uint32)
                    payload = values.tobytes()
                    stream.write(payload)
                    digest.update(payload)
                    written += len(values)
                    documents += 1
                if documents % 4096 < 256:
                    print(f"{model}: {written:,} tokens / {documents:,} documents", flush=True)
                if written >= args.tokens:
                    break
        if written < args.tokens:
            raise RuntimeError(f"Source prefix produced only {written} tokens")
        destination.replace(root / "train.bin")
        (root / "meta.json").write_text(json.dumps({
            "dataset": source_meta["dataset"], "source": str(args.source),
            "preparation": "decode GPT-2 documents, re-encode with official Qwen tokenizer",
            "tokenizer": provenance["repository"], "revision": provenance["revision"],
            "dtype": "uint32", "tokens": written, "documents": documents,
            "eos_id": tokenizer.eos_token_id, "vocab_size": len(tokenizer),
            "sha256": digest.hexdigest(),
        }, indent=2) + "\n")
        print(f"DONE {root}: {written:,} tokens", flush=True)


if __name__ == "__main__":
    main()
