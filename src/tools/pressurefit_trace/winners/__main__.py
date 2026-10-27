"""Prepare saved quickstart winners for tools.pressurefit_trace.matrix render."""

import argparse
from pathlib import Path

from .prepare import prepare


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("quickstart", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--workload")
    args = parser.parse_args(argv)
    prepare(args.quickstart, args.output, rank=args.rank, workload=args.workload)


if __name__ == "__main__":
    main()
