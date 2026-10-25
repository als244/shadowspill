"""Qwen3.5 MoE LoRA through separate build/profile/plan/execute workers."""

import argparse
import json
from pathlib import Path

from workloads.recipes.text.phases import run_phases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="Qwen3.5-tokenized corpus directory")
    parser.add_argument("--checkpoint", help="Local Hugging Face base checkpoint")
    parser.add_argument("--frozen-state-id", help="Immutable base and config identity")
    parser.add_argument("--run-dir", default="runs/qwen35-lora")
    parser.add_argument(
        "--phase", choices=["all", "build", "profile", "plan", "execute"], default="all"
    )
    parser.add_argument("--sequence-length", type=int, default=32768)
    parser.add_argument("--sequences-per-step", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=float, default=32)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--execution-gib", type=float)
    parser.add_argument("--spill-gib", type=float)
    parser.add_argument("--warmup-seconds", type=float, default=0.05)
    parser.add_argument("--measure-seconds", type=float, default=0.18)
    parser.add_argument("--no-plots", dest="plots", action="store_false")
    parser.add_argument("--resume", help="Saved Trainer checkpoint archive")
    parser.add_argument("--wandb-project")
    args = parser.parse_args(argv)
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
        if args.phase not in {"all", "build"} or not all(
            (args.data, args.checkpoint, args.frozen_state_id)
        ):
            parser.error("start with --data, --checkpoint and --frozen-state-id")
        options = vars(args).copy()
        for key in ("data", "checkpoint"):
            options[key] = str(Path(options[key]).resolve())
        options["execution_gib"] = (
            28 if args.execution_gib is None else args.execution_gib
        )
        options["spill_gib"] = 144 if args.spill_gib is None else args.spill_gib
    if (
        min(
            options[key]
            for key in (
                "sequence_length",
                "sequences_per_step",
                "steps",
                "lora_rank",
                "execution_gib",
                "spill_gib",
            )
        )
        <= 0
    ):
        parser.error("shapes, steps, rank and budgets must be positive")
    root.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(options, indent=2) + "\n")
    run_phases("workloads.recipes.text.qwen35_lora", root, args.phase)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
