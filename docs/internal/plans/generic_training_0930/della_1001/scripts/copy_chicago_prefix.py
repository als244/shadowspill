"""Copy a document-complete prefix of Chicago's prepared GPT-2 token stream.

Run on the head node. The active planning sample is left untouched.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=2_000_000_000)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("/home/as1669/storage/datasets/fineweb_edu_gpt2_chicago_2b"),
    )
    args = parser.parse_args()
    if args.tokens <= 0:
        raise ValueError("--tokens must be positive")
    sample = args.outdir.with_name("fineweb_edu_gpt2_chicago_sample")
    target = args.outdir
    target.mkdir(parents=True, exist_ok=True)
    if (target / "meta.json").exists():
        raise FileExistsError(f"Completed data already exists: {target}")
    partial = target / "train.bin.partial"
    byte_count = 4 * args.tokens
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "chicago",
        f"head -c {byte_count} "
        "/home/shein/Documents/datasets/fineweb_edu_gpt2_10b/train.bin",
    ]
    print(f"Copying {args.tokens:,} token slots to {partial}", flush=True)
    with partial.open("wb") as output:
        subprocess.run(command, stdout=output, check=True)
    if partial.stat().st_size != byte_count:
        raise ValueError("Source stream ended before the requested prefix")
    meta = json.loads((sample / "meta.json").read_text())
    tail_bytes = min(byte_count, 4 * 1024 * 1024)
    with partial.open("r+b") as stream:
        stream.seek(byte_count - tail_bytes)
        tail = np.frombuffer(stream.read(), dtype=np.uint32)
        ends = np.flatnonzero(tail == meta["eos_id"])
        if not len(ends):
            raise ValueError("No document boundary in the final million tokens")
        final_bytes = byte_count - tail_bytes + 4 * (int(ends[-1]) + 1)
        stream.truncate(final_bytes)
    with partial.open("rb") as copied, (sample / "train.bin").open("rb") as original:
        while chunk := original.read(16 * 1024 * 1024):
            if copied.read(len(chunk)) != chunk:
                raise ValueError(
                    "Copied prefix does not match the existing planning sample"
                )
    with partial.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    partial.replace(target / "train.bin")
    shutil.copy2(sample / "val.bin", target / "val.bin")
    meta.update(
        train_tokens=final_bytes // 4,
        copied_utc=datetime.now(UTC).isoformat(),
        subset_requested_tokens=args.tokens,
        train_sha256=digest,
        matches_existing_planning_prefix=True,
    )
    (target / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        json.dumps(
            {
                "directory": str(target),
                "train_tokens": meta["train_tokens"],
                "sha256": digest,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
