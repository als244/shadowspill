"""The Hugging Face transformers loss on the documents a run's step trained on.

    python -m training.reference <run dir> [--checkpoint <HF directory>] \\
        [--step 1] [--device cuda|cpu|auto] [--dtype bfloat16|float32] [--tag NAME]

Rebuilds the step's documents from the run's packing records and its token
files (under ``long_documents=truncate`` every document is one sequence,
indexed by its ordinal), runs each through the checkpoint in transformers,
sums the next-token cross-entropy over the same targets, the mask included,
and divides by the trained-token count, which is the runner's own
normalization. Prints that loss beside the run's recorded loss and writes
``reference[-tag].json`` in the run. The checkpoint defaults to the directory
the run's model was built from. ``auto`` places a checkpoint larger than the
card in host memory and visits the GPU layer by layer.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from workloads.recipes.text.data.documents import IGNORE, TokenDocuments


def load_model(directory: str, device: str, dtype: torch.dtype) -> Any:
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {"dtype": dtype, "attn_implementation": "sdpa"}
    if device == "auto":
        kwargs.update(device_map="auto", max_memory={0: "24GiB", "cpu": "150GiB"})
    try:
        model = AutoModelForCausalLM.from_pretrained(directory, **kwargs)
    except ValueError:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(directory, **kwargs)
    if device != "auto":
        model.to(device)
    model.eval()
    return model


def step_record(run: Path, step: int) -> dict[str, Any]:
    for line in (run / "packing.jsonl").read_text().splitlines():
        record = json.loads(line)
        if record["step"] == step:
            return record
    raise ValueError(f"no packing record for step {step} in {run}")


def recorded_loss(run: Path, step: int) -> float | None:
    for line in (run / "metrics.jsonl").read_text().splitlines():
        record = json.loads(line)
        if record.get("step") == step and "train/loss" in record:
            return float(record["train/loss"])
    return None


def reference_loss(
    run: Path,
    *,
    checkpoint: str | None = None,
    step: int = 1,
    device: str = "cuda",
    dtype: str = "bfloat16",
    tag: str = "",
) -> dict[str, Any]:
    """The transformers loss on the step's documents beside the run's, written
    to ``reference[-tag].json`` in the run and returned."""

    options = json.loads((run / "config.json").read_text())
    data = options["data"]
    if data.get("long_documents", "drop") != "truncate":
        raise ValueError("document ordinals index sequences only under truncate")
    if checkpoint is None:
        checkpoint = options.get("model", {}).get("directory")
        if checkpoint is None:
            raise ValueError("the run names no checkpoint directory; pass one")
    directory = Path(data["directory"])
    meta = json.loads((directory / "meta.json").read_text())
    mask = directory / "train.mask.bin"
    documents = TokenDocuments(
        directory / "train.bin",
        int(meta["eos_id"]),
        int(options["max_seq_len"]),
        "truncate",
        mask=mask if mask.exists() else None,
    )
    record = step_record(run, step)
    model = load_model(checkpoint, device, getattr(torch, dtype))
    compute = "cuda" if device == "auto" else device
    total, trained, count = 0.0, 0, 0
    with torch.no_grad():
        for microbatch in record["microbatches"]:
            for entry in microbatch:
                ordinal = int(entry[0])
                tokens = torch.from_numpy(documents.tokens(ordinal).astype("int64"))
                targets = torch.from_numpy(documents.targets(ordinal))
                logits = model(input_ids=tokens[None].to(compute)).logits[0]
                total += float(
                    F.cross_entropy(
                        logits.float(),
                        targets.to(logits.device),
                        ignore_index=IGNORE,
                        reduction="sum",
                    )
                )
                trained += int((targets != IGNORE).sum())
                count += 1
    reference = total / trained
    recorded = recorded_loss(run, step)
    result = {
        "step": step,
        "documents": count,
        "trained_tokens": trained,
        "recorded_trained_tokens": record.get("trained_tokens"),
        "transformers_loss": reference,
        "recorded_loss": recorded,
        "difference": None if recorded is None else recorded - reference,
        "relative": None
        if recorded is None
        else (recorded - reference) / max(abs(reference), 1e-12),
        "checkpoint": str(checkpoint),
        "dtype": dtype,
        "device": device,
    }
    (run / f"reference{'-' + tag if tag else ''}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--checkpoint", help="defaults to the run's model directory")
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--tag", default="", help="suffix of the result file")
    args = parser.parse_args(argv)
    result = reference_loss(
        args.run,
        checkpoint=args.checkpoint,
        step=args.step,
        device=args.device,
        dtype=args.dtype,
        tag=args.tag,
    )
    print(json.dumps(result, indent=2))
    recorded = result["recorded_loss"]
    if recorded is not None and not math.isfinite(recorded):
        raise SystemExit("the run's recorded loss is not finite")


if __name__ == "__main__":
    main()
