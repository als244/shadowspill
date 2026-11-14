"""Plan saved candidates under different host-transfer rates, without a GPU."""

import argparse
import math
from pathlib import Path

from shadowspill.planner import named_resolution_options


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("expected a finite positive number")
    return number


def pairs(value):
    try:
        result = [
            tuple(positive(v) for v in pair.split("/")) for pair in value.split(",")
        ]
        if not result or any(len(pair) != 2 for pair in result):
            raise ValueError
        return result
    except (ValueError, argparse.ArgumentTypeError) as error:
        raise argparse.ArgumentTypeError(
            "use FETCH/EVICT pairs, e.g. 25/25,100/50"
        ) from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", action="append", metavar="NAME=PROFILED.ssp")
    parser.add_argument("--budget-gib", type=positive, nargs="+")
    parser.add_argument("--spill-gib", type=positive, nargs="+", default=[112.0])
    parser.add_argument("--sim-transfer-bandwidths", type=pairs)
    parser.add_argument("--communication-info", type=Path)
    parser.add_argument("--latency-us", type=float, nargs=2, default=None)
    parser.add_argument(
        "--orderings", choices=("factors", "depth-first"), default="factors"
    )
    parser.add_argument("--resolution-options", type=named_resolution_options)
    parser.add_argument("--search-workers", type=int, default=1)
    parser.add_argument("--units-per-step", type=positive, default=1.0)
    parser.add_argument("--unit-label", default="steps")
    parser.add_argument("--title")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--replot", type=Path)
    parser.add_argument("--measured", type=Path)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if args.replot is not None:
        from .plot import draw

        draw(args.replot, args.output_dir, title=args.title)
        return 0
    if not args.program or not args.budget_gib or not args.sim_transfer_bandwidths:
        parser.error(
            "planning requires --program, --budget-gib and --sim-transfer-bandwidths"
        )
    if args.search_workers < 1:
        parser.error("--search-workers must be positive")
    if args.latency_us is not None and any(
        not math.isfinite(v) or v < 0 for v in args.latency_us
    ):
        parser.error("--latency-us must contain finite nonnegative values")
    programs = {}
    for value in args.program:
        name, separator, path = value.partition("=")
        if not separator or not name or not path or name in programs:
            parser.error("--program needs a unique NAME=PROFILED.ssp")
        programs[name] = Path(path).resolve()
    from .run import run

    run(args, programs)
    return 0
