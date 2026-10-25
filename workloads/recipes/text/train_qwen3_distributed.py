"""Qwen3 MoE full training through independently runnable public phases.

Run this once; communicate/execute launch their own torchrun workers.
Defaults: eight GPUs, 4K tokens, 64 sequences per rank (512 globally).
"""

import argparse
import json
from pathlib import Path

from workloads.recipes.phases import run_phases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="Qwen3-tokenized corpus directory")
    parser.add_argument("--run-dir", type=Path, default=Path("runs/qwen3-dp8"))
    parser.add_argument(
        "--phase",
        default="all",
        choices=["all", "build", "profile", "communicate", "plan", "execute"],
    )
    parser.add_argument(
        "--model",
        choices=["qwen3moe-tiny", "qwen3moe-30b-a3b"],
        default="qwen3moe-30b-a3b",
    )
    parser.add_argument("--ranks", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument(
        "--sequences-per-step",
        type=int,
        default=64,
        help="number of sequences per rank in one update",
    )
    parser.add_argument("--max-sequences-per-example", type=int, default=32)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--execution-gib", type=float)
    parser.add_argument("--spill-gib", type=float)
    parser.add_argument("--warmup-seconds", type=float, default=0.05)
    parser.add_argument("--measure-seconds", type=float, default=0.18)
    parser.add_argument("--no-plots", dest="plots", action="store_false")
    args = parser.parse_args(argv)
    root = args.run_dir.resolve()
    config = root / "config.json"
    if args.phase in {"all", "build"}:
        if config.exists() or not args.data:
            parser.error("start with --data and a fresh run directory")
        options = vars(args).copy()
        options["run_dir"] = str(root)
        options["data"] = str(Path(args.data).resolve())
        options["execution_gib"] = (
            77 if args.execution_gib is None else args.execution_gib
        )
        options["spill_gib"] = 96 if args.spill_gib is None else args.spill_gib
    else:
        if not config.exists():
            parser.error("run the build phase first")
        options = json.loads(config.read_text())
        for key in ("execution_gib", "spill_gib"):
            if getattr(args, key) is not None:
                if args.phase != "plan":
                    parser.error("changed budgets require --phase plan")
                options[key] = getattr(args, key)
    if (
        min(
            options[key]
            for key in (
                "ranks",
                "sequence_length",
                "sequences_per_step",
                "max_sequences_per_example",
                "steps",
                "learning_rate",
                "execution_gib",
                "spill_gib",
            )
        )
        <= 0
    ):
        parser.error("counts, learning rate and memory budgets must be positive")
    if args.phase == "communicate" and options["ranks"] == 1:
        parser.error("single-rank host-copy calibration is included in --phase profile")
    root.mkdir(parents=True, exist_ok=True)
    config.write_text(json.dumps(options, indent=2) + "\n")
    run_phases("workloads.recipes.text.qwen3", root, args.phase, ranks=options["ranks"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
