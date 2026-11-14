"""Collection commands: saved profiles, CPU search, decisions and packed export."""

import argparse
from pathlib import Path

from .budgets import transfer_bandwidths
from .dump import dump
from .export import export
from .planning import plan
from .record import trace


def add_commands(modes):
    dumping = modes.add_parser(
        "dump", help="Export saved quickstart profiles as diagnostic planner inputs"
    )
    dumping.add_argument("input", type=Path, help="Completed phase profiling directory")
    dumping.add_argument("--output", type=Path, required=True)
    dumping.add_argument(
        "--physical-gib",
        type=float,
        help="Export cap including overhead (default: largest requested budget)",
    )
    dumping.set_defaults(run=dump)
    planning = modes.add_parser(
        "plan", help="Plan every captured program without an accelerator"
    )
    planning.add_argument("input", type=Path)
    planning.add_argument("--output", type=Path, required=True)
    budgets = planning.add_mutually_exclusive_group()
    budgets.add_argument(
        "--execution-gib", type=float, help="Usable planner slab, excluding overhead"
    )
    budgets.add_argument(
        "--physical-gib",
        type=float,
        help="Physical device cap; subtract the captured runtime/headroom carve-out",
    )
    planning.add_argument("--spill-gib", type=float)
    planning.add_argument(
        "--transfer-bandwidths",
        type=transfer_bandwidths,
        help="Override decimal GB/s: FETCH,EVICT or "
        "FETCH_SOLO,FETCH_CONCURRENT,EVICT_SOLO,EVICT_CONCURRENT; "
        "retain captured task profiles and transfer latencies",
    )
    planning.add_argument("--workers", type=int, default=0)
    planning.set_defaults(run=plan)
    tracing = modes.add_parser(
        "trace", help="Record the winner, explicit pairs, or all candidates"
    )
    tracing.add_argument("input", type=Path)
    tracing.add_argument("--output", type=Path, required=True)
    selection = tracing.add_mutually_exclusive_group()
    selection.add_argument(
        "--candidates-json",
        type=Path,
        help="JSON list of [selection_id, candidate_id] pairs",
    )
    tracing.add_argument("--programs", choices=("winner", "all"), default="winner")
    selection.add_argument(
        "--all-candidates",
        action="store_true",
        help="Record every resolution x policy in each selected program",
    )
    tracing.set_defaults(run=trace)
    exporting = modes.add_parser(
        "export", help="Verify and export traces to annotated JSON / NumPy"
    )
    exporting.add_argument("input", type=Path)
    exporting.add_argument("--output", type=Path, required=True)
    exporting.set_defaults(run=export)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    add_commands(modes)
    args = parser.parse_args()
    args.run(args)
