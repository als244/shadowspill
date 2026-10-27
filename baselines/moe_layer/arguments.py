"""Arguments for one router/expert forward-backward measurement."""

import argparse
from pathlib import Path

from .config import add_shape_arguments
from .implementations.registry import BACKENDS


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=list(BACKENDS), required=True)
    parser.add_argument("--precision", choices=["bf16", "fp8_current"], required=True)
    parser.add_argument("--training", choices=["full", "lora"], required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument(
        "--tiny", action="store_true", help="small qualification geometry"
    )
    parser.add_argument(
        "--sonicmoe-candidate-policy",
        choices=["baseline", "expanded", "expanded-k"],
        default="baseline",
    )
    add_shape_arguments(parser)
    args = parser.parse_args(argv)
    if min(args.tokens, args.warmup, args.iterations, args.cpu_threads) < 1:
        parser.error("token, step and thread counts must be positive")
    return args
