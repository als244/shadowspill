"""Optional command-line spelling of the shared Qwen3 configuration."""

import argparse
from pathlib import Path


def parser():
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
    parser.add_argument(
        "--microbatch-sizes",
        type=int,
        nargs="+",
        help="Exact sequences-per-microbatch candidates; each must divide "
        "sequences-per-step; default: 1 2 4 8 16 32",
    )
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--schedule-steps", type=int, default=10)
    parser.add_argument("--parallelism", choices=["dp", "ep"], default="dp")
    parser.add_argument("--resume")
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[0])
    parser.add_argument("--moe-chunk-sizes", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--graph-pair-policies",
        nargs="+",
        default=["cost-input"],
        choices=["cost-input", "cost-attention", "attention-input"],
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--execution-gib", type=float)
    parser.add_argument("--spill-gib", type=float)
    parser.add_argument("--external-headroom-mib", type=int, default=768)
    parser.add_argument("--warmup-seconds", type=float, default=0.05)
    parser.add_argument("--measure-seconds", type=float, default=0.18)
    parser.add_argument("--no-plots", dest="plots", action="store_false")
    return parser
