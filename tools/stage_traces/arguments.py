"""Command-line settings for the isolated-stage trace sweep."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory", required=True)
    parser.add_argument("--factory-args", type=json.loads, default={})
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-lengths", type=int, nargs="+", default=[8192])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["save", "recompute"],
        default=["save", "recompute"],
    )
    parser.add_argument("--gpus", nargs="+", default=["all"])
    parser.add_argument("--nsys", default="nsys")
    parser.add_argument(
        "--gpu-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="record hardware metrics on the worker's assigned GPU (default: on)",
    )
    parser.add_argument("--gpu-metrics-frequency", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--discovery-timeout", type=int, default=3600)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--discover-only", action="store_true")
    args = parser.parse_args()
    if (
        min(
            *args.sequence_lengths,
            *args.batch_sizes,
            args.warmup,
            args.iterations,
            args.cpu_threads,
            args.timeout,
        )
        <= 0
    ):
        parser.error("geometry and measurement counts must be positive")
    if not 10 <= args.gpu_metrics_frequency <= 200000:
        parser.error("GPU metric frequency must be between 10 and 200000 Hz")
    return args
