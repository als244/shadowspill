"""Performance matrix selection and command-line options."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from qualification.device_defaults import add_memory_budget_arguments
from qualification.performance.cases import FullModelManifest, manifests
from qualification.precision import add_dtype_arguments


def parse_cell_budgets(
    entries: Sequence[str],
    identities: frozenset[str],
    *,
    flag: str = "planning-spill-budget-gib",
) -> dict[str, int]:
    """Parse repeatable ``IDENTITY=GIB`` per-cell budget overrides."""

    budgets: dict[str, int] = {}
    for entry in entries:
        identity, separator, gib_text = entry.partition("=")
        if not separator:
            raise ValueError(f"{flag} entry {entry!r} must be IDENTITY=GIB")
        if identity not in identities:
            raise ValueError(f"{flag} names unknown cell {identity!r}")
        if identity in budgets:
            raise ValueError(f"{flag} repeats cell {identity!r}")
        gib = int(gib_text)
        if gib <= 0:
            raise ValueError(f"{flag} for {identity!r} must be positive")
        budgets[identity] = gib
    return budgets


def default_cells() -> tuple[FullModelManifest, ...]:
    """The cells the gate runs when none are named.

    Preserve the established three mlops workloads. Throughput floors are
    scoped at execution time; runtime, memory and simulator checks apply to
    every selected cell, including explicit pure-PyTorch selections.
    """

    return tuple(
        item for item in manifests() if item.regression_tokens_per_second is not None
    )


def create_parser() -> argparse.ArgumentParser:
    """Which cells to run, in which protocol, and where to put what they write."""

    parser = argparse.ArgumentParser(
        description=(
            "Launch the ShadowSpill-only full-model qualification cells.\n\n"
            "The gate runs the three mlops cells by default; ``--cells`` also "
            "selects the\npure-PyTorch variants. Every cell checks runtime, "
            "physical budgets and simulator\naccuracy. Throughput comparisons "
            "additionally require a matching baseline."
        )
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("qualification/results/full_model"),
    )
    parser.add_argument(
        "--execution-budget-gib",
        type=int,
        help="total GPU allowance in GiB (default: 16; 10 below SM80)",
    )
    add_dtype_arguments(parser)
    add_memory_budget_arguments(parser)
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument(
        "--empty-caches",
        action="store_true",
        help=(
            "start every cell with empty Inductor "
            "and Triton caches, in a temporary directory removed afterwards, so "
            "nothing is reused from an earlier run or another cell"
        ),
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--measure-only",
        action="store_true",
        help=(
            "report each cell's throughput without judging it, and exit on "
            "whether the cells ran rather than on whether they passed. "
            "Hardware/configuration mismatches already exclude unrelated "
            "throughput floors without this flag"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        action="store_true",
        help=(
            "opt into streamed checkpoint/restore; "
            "the matrix default skips checkpoint I/O"
        ),
    )
    parser.add_argument(
        "--cells",
        nargs="*",
        help=(
            "identities to run, such as mlops_llama3 or pytorch_qwen35; "
            "defaults to every cell carrying a throughput authority"
        ),
    )
    parser.add_argument(
        "--planning-spill-budget-gib",
        action="append",
        default=[],
        metavar="IDENTITY=GIB",
        help=(
            "per-cell planning spill budget within the configured pool, "
            "for example mlops_qwen35=100; repeatable"
        ),
    )
    parser.add_argument(
        "--spill-budget-gib",
        action="append",
        default=[],
        metavar="IDENTITY=GIB",
        help=(
            "per-cell spill-pool capacity in place of the manifest's, for a "
            "host with less memory than the manifests were sized on, for "
            "example mlops_llama3=80; repeatable"
        ),
    )
    return parser
