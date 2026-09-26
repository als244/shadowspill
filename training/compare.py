"""Compare runs' loss curves step by step.

    python -m training.compare <reference run dir> <run dir> [<run dir> ...]

Every run trained on the same batches from the same initial weights, so the
difference at each step is only what the runs computed differently -- which
each run's packing.jsonl lets this check rather than assume. Prints, for each
run against the first: whether they trained on the same documents, the step-0
difference, the largest over the first 10 steps, the mean over each 100
steps, and the final validation loss; then draws the curves and differences
into the reference run's compare.png.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(run: Path) -> tuple[dict[int, float], dict[int, float]]:
    """Training and validation loss by step; a resumed run's later lines win."""

    train, val = {}, {}
    for line in (run / "metrics.jsonl").read_text().splitlines():
        record = json.loads(line)
        if "loss" in record:
            train[record["step"]] = record["loss"]
        if "val_loss" in record:
            val[record["step"]] = record["val_loss"]
    return train, val


def packing(run: Path) -> dict[int, list]:
    """The documents each step trained on; a resumed run's later lines win."""

    lines = (run / "packing.jsonl").read_text().splitlines()
    return {record["step"]: record["microbatches"] for record in map(json.loads, lines)}


def main() -> None:
    runs = [Path(path) for path in sys.argv[1:]]
    reference, *others = runs
    ref_train, ref_val = read(reference)
    figure, (curves, gaps) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    # One color per run, the same in both panels: C0 is the reference.
    curves.plot(
        list(ref_train),
        list(ref_train.values()),
        label=reference.name,
        lw=0.8,
        color="C0",
    )

    print(
        f"reference: {reference.name}, final val loss {list(ref_val.values())[-1]:.4f}"
    )
    for index, run in enumerate(others, start=1):
        train, val = read(run)
        steps = sorted(set(train) & set(ref_train))
        gap = {step: train[step] - ref_train[step] for step in steps}
        windows = [
            sum(abs(gap[s]) for s in steps[i : i + 100]) / len(steps[i : i + 100])
            for i in range(0, len(steps), 100)
        ]
        print(f"\n{run.name}: {len(steps)} shared steps")
        ours, theirs = packing(run), packing(reference)
        differing = [
            step
            for step in sorted(set(ours) & set(theirs))
            if ours[step] != theirs[step]
        ]
        print(
            "  documents        "
            + (
                f"DIFFER from step {differing[0]}"
                if differing
                else "identical at every step"
            )
        )
        first = f"{train[0]:.6f} vs {ref_train[0]:.6f}"
        print(f"  step 0 loss      {first} (diff {gap[0]:+.2e})")
        print(f"  max |diff| 0-9   {max(abs(gap[s]) for s in steps[:10]):.2e}")
        print("  mean |diff| / 100 steps  " + "  ".join(f"{w:.4f}" for w in windows))
        if val:
            final = max(set(val) & set(ref_val))
            difference = val[final] - ref_val[final]
            print(f"  val loss at {final}  {val[final]:.4f} (diff {difference:+.4f})")
        curves.plot(
            steps, [train[s] for s in steps], label=run.name, lw=0.8, color=f"C{index}"
        )
        gaps.plot(steps, list(gap.values()), label=run.name, lw=0.6, color=f"C{index}")

    curves.set_ylabel("training loss")
    curves.legend()
    gaps.axhline(0, color="black", lw=0.5)
    gaps.legend()
    gaps.set_ylabel(f"loss minus {reference.name}")
    gaps.set_xlabel("step")
    figure.tight_layout()
    figure.savefig(reference / "compare.png", dpi=120)
    print(f"\nwrote {reference / 'compare.png'}")


if __name__ == "__main__":
    main()
