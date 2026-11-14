"""Capture Nsight reports from completed phase quickstarts without rebuilding."""

import argparse
import math
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("quickstart", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nsys", default="nsys")
    parser.add_argument(
        "--world-size", type=int, help="optional check against saved worker count"
    )
    parser.add_argument("--budget-gib", type=float)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--capture-steps", type=int, default=3)
    parser.add_argument(
        "--capture-settle-seconds",
        type=float,
        default=1.0,
        help="idle at each capture edge, outside annotated steps",
    )
    parser.add_argument("--gpu-metrics-frequency", type=int, default=10000)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--plan-digest", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.quickstart, args.output = args.quickstart.resolve(), args.output.resolve()
    if (
        min(
            1 if args.world_size is None else args.world_size,
            args.warmup_steps,
            args.capture_steps,
            args.gpu_metrics_frequency,
        )
        < 1
    ):
        parser.error("world size, step counts and metric frequency must be positive")
    if (
        not math.isfinite(args.capture_settle_seconds)
        or args.capture_settle_seconds < 0
    ):
        parser.error("capture settling interval must be finite and nonnegative")
    if args.budget_gib is not None and (
        not math.isfinite(args.budget_gib) or args.budget_gib <= 0
    ):
        parser.error("budget must be finite and positive")
    if args.worker:
        if args.budget_gib is None or args.plan_digest is None:
            parser.error("worker requires one budget and its expected Plan digest")
        from .worker import run

        return run(args)
    from .capture import capture

    return capture(args)


if __name__ == "__main__":
    raise SystemExit(main())
