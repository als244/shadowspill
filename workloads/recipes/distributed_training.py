"""Sharded-DP regression: independent build/profile/communication/plan/execute."""

import argparse
import json
from pathlib import Path

from workloads.recipes.phases import run_phases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=["all", "build", "profile", "communicate", "plan", "execute"],
        default="all",
    )
    parser.add_argument("--ranks", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--execution-gib", type=float, default=2)
    parser.add_argument("--spill-gib", type=float, default=1)
    parser.add_argument("--warmup-seconds", type=float, default=0.05)
    parser.add_argument("--measure-seconds", type=float, default=0.18)
    parser.add_argument("--no-plots", dest="plots", action="store_false")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    args = parser.parse_args(argv)
    root = args.run_dir.resolve()
    config = root / "config.json"
    if args.phase in {"all", "build"}:
        if config.exists():
            parser.error("use a fresh directory for a new build")
        if args.ranks < 2 or args.steps < 1:
            parser.error("use at least two ranks and a positive step count")
        if min(args.execution_gib, args.spill_gib) <= 0:
            parser.error("memory budgets must be positive")
        options = vars(args).copy()
        options["run_dir"] = str(root)
        root.mkdir(parents=True, exist_ok=True)
        config.write_text(json.dumps(options, indent=2) + "\n")
    else:
        if not config.exists():
            parser.error("run the build phase first")
        options = json.loads(config.read_text())
    run_phases("workloads.recipes.regression", root, args.phase, ranks=options["ranks"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
