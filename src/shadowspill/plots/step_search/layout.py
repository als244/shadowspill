"""Where a search's figures are written: one directory per question they answer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FigureTree:
    """The directories one search's figures and tables are written into.

    Everything sits under ``sim`` because every figure here reads a plan
    rather than a run; ``real`` is its counterpart, written by the run
    figures. ``root`` is above both, and holds the raw data the whole set
    was drawn from.
    """

    root: Path
    sim: Path
    throughput: Path
    overheads: Path
    transfers: Path
    unconstrained: Path
    by_selection: Path
    lanes_by_selection: Path
    orderings: Path

    @classmethod
    def under(cls, directory: str | Path) -> FigureTree:
        """Make every directory the family writes into and return them."""

        root = Path(directory)
        sim = root / "sim"
        overheads = sim / "overheads"
        transfers = sim / "transfers"
        tree = cls(
            root=root,
            sim=sim,
            throughput=sim / "throughput",
            overheads=overheads,
            transfers=transfers,
            unconstrained=sim / "vs_unconstrained",
            by_selection=overheads / "by_graph_pair_selection",
            lanes_by_selection=transfers / "by_graph_pair_selection",
            orderings=sim / "orderings",
        )
        for path in (
            tree.sim,
            tree.throughput,
            tree.overheads,
            tree.transfers,
            tree.unconstrained,
            tree.by_selection,
            tree.lanes_by_selection,
            tree.orderings,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return tree
