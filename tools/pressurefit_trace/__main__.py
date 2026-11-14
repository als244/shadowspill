"""Capture and analyze recorded PressureFit decisions."""

import argparse
import json
from pathlib import Path

from .collect.cli import add_commands
from .data import Trajectory, analyze


def run_analysis(args):
    data = Trajectory.load(args.input, args.candidate)
    report = analyze(data, args.output)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "candidate_index",
                    "compute_seconds",
                    "row_filter",
                    "reductions",
                )
            },
            indent=2,
        )
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    add_commands(modes)
    analysis = modes.add_parser(
        "analyze", help="Verify a candidate and export cut analysis"
    )
    analysis.add_argument(
        "input", type=Path, help="Raw trace directory containing manifest.json"
    )
    analysis.add_argument(
        "--candidate", default="winner", help="winner or zero-based candidate index"
    )
    analysis.add_argument("--output", type=Path, required=True)
    analysis.set_defaults(run=run_analysis)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.run(args)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
