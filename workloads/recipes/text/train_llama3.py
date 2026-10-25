"""Llama 3 full training with independently runnable build/profile/plan/execute."""

import argparse
import json
from pathlib import Path

from workloads.recipes.text.phases import run_phases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", help="Llama-tokenized corpus: train/val.bin, meta.json"
    )
    parser.add_argument("--run-dir", default="runs/llama3-8b")
    parser.add_argument(
        "--phase", choices=["all", "build", "profile", "plan", "execute"], default="all"
    )
    parser.add_argument(
        "--model",
        choices=["llama3-tiny", "llama3-1b", "llama3-8b"],
        default="llama3-8b",
    )
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--sequences-per-step", type=int, default=16)
    parser.add_argument("--sequences-per-example", type=int, default=4)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--execution-gib", type=float)
    parser.add_argument("--spill-gib", type=float)
    parser.add_argument("--warmup-seconds", type=float, default=0.05)
    parser.add_argument("--measure-seconds", type=float, default=0.18)
    parser.add_argument("--no-plots", dest="plots", action="store_false")
    parser.add_argument("--resume", help="checkpoint for the execute phase")
    args = parser.parse_args(argv)
    if args.steps < 1 or args.checkpoint_every < 0:
        parser.error("steps must be positive and checkpoint cadence nonnegative")
    root = Path(args.run_dir).resolve()
    config_path = root / "config.json"
    if config_path.exists():
        if args.phase in {"all", "build"}:
            parser.error("use a fresh run directory for a new build")
        options = json.loads(config_path.read_text())
        for key in ("execution_gib", "spill_gib", "resume"):
            if getattr(args, key) is not None:
                if key != "resume" and args.phase != "plan":
                    parser.error("changed budgets require --phase plan")
                options[key] = getattr(args, key)
    else:
        if args.phase not in {"all", "build"} or not args.data:
            parser.error("start with --data and --phase all or build")
        options = vars(args).copy()
        options["data"] = str(Path(args.data).resolve())
        options["execution_gib"] = (
            28 if args.execution_gib is None else args.execution_gib
        )
        options["spill_gib"] = 144 if args.spill_gib is None else args.spill_gib
    if min(options["execution_gib"], options["spill_gib"]) <= 0:
        parser.error("memory budgets must be positive")
    for name in ("logs", "training"):
        (root / name).mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(options, indent=2) + "\n")
    run_phases("workloads.recipes.text.llama3", root, args.phase)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
