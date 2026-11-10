"""Optional CLI spelling of the public Qwen35 configuration."""

import argparse


def parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="Qwen3.5-tokenized corpus directory")
    parser.add_argument("--checkpoint", help="Local Hugging Face base checkpoint")
    parser.add_argument("--frozen-state-id", help="Immutable base and config identity")
    parser.add_argument("--run-dir", default="runs/qwen35-lora")
    parser.add_argument(
        "--phase",
        choices=["all", "build", "profile", "communicate", "plan", "execute"],
        default="all",
    )
    parser.add_argument("--sequence-length", type=int, default=32768)
    parser.add_argument("--sequences-per-step", type=int, default=8)
    parser.add_argument(
        "--microbatch-sizes",
        type=int,
        nargs="+",
        help="Exact sequences-per-microbatch candidates; each must divide "
        "sequences-per-step; default: 1 2 4 8",
    )
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=float, default=32)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--schedule-steps", type=int, default=10)
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[0])
    parser.add_argument("--moe-chunk-sizes", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--graph-pair-policies",
        nargs="+",
        default=["cost-input"],
        choices=["cost-input", "cost-attention", "attention-input"],
    )
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--eval-updates", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=2)
    parser.add_argument("--external-headroom-mib", type=int, default=512)
    parser.add_argument("--search-workers", type=int, default=0)
    parser.add_argument("--resolution-options", nargs="+")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--execution-gib", type=float)
    parser.add_argument("--spill-gib", type=float)
    parser.add_argument("--warmup-seconds", type=float, default=0.05)
    parser.add_argument("--measure-seconds", type=float, default=0.18)
    parser.add_argument("--no-plots", dest="plots", action="store_false")
    parser.add_argument("--resume", help="Saved Trainer checkpoint archive")
    parser.add_argument("--wandb-project")
    return parser
