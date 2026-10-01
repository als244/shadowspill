"""Durable quickstart output directories and wall-time accounting."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from .options import _request_record, _revision, _started_at


@dataclass(frozen=True)
class RunPaths:
    """Where one run writes, and the stores it reads and writes."""

    root: Path
    store: Path
    build_store: Path | None
    plan_store: Path


def default_run_root(arguments: argparse.Namespace, request) -> Path:
    return arguments.output_dir or (
        Path("benchmarking/quickstart_reports")
        / f"{request.label}_{_revision()}_{_started_at()}"
    )


def prepare_run_root(
    arguments: argparse.Namespace, request, *, rank: int | None = None
) -> RunPaths | None:
    """Claim the run directory, record the request, copy the console into it.

    `None` when the directory already holds a run and `--force-overwrite` was
    not given; the refusal has been printed.
    """

    # Everything a run leaves behind lands together: the search report, its
    # log, and one step trace per run budget.
    # Each run gets its experiment label, revision and start time. Shape and
    # data geometry belong to the factory and are recorded in request.json.
    base = default_run_root(arguments, request)
    suffix = None if rank is None else f"rank-{rank:05d}"
    run_root = base if suffix is None else base / suffix
    # A run directory is written once, and silently replacing one loses a
    # measurement that cost real time. The default path carries the start
    # minute, so this guards an explicit --output-dir and the two runs that
    # begin within the same minute. Refuse, and say both ways out.
    written = tuple(
        name
        for name in (
            "search.json",
            "progress.log",
            "steps",
            "figures",
            "timelines",
            "step_metrics.jsonl",
        )
        if (run_root / name).exists()
    )
    if written and not arguments.force_overwrite:
        print(
            f"  {run_root} already holds a run ({', '.join(written)}).\n"
            "  Pass --force-overwrite to replace it, or --output-dir to write"
            " somewhere else.",
            file=sys.stderr,
        )
        return None
    for name in written:
        target = run_root / name
        # The artifact store is deliberately not cleared: it is a
        # content-addressed cache, so stale entries are unreachable rather
        # than wrong, and rebuilding it costs capture, compilation and
        # profiling over again.
        shutil.rmtree(target) if target.is_dir() else target.unlink()

    # A run owns both stores by default, so what it measured is self-contained
    # and nothing it reused is ambiguous. Point `--artifact-store` at
    # another run's store, or at a shared one, to skip capture, compilation
    # and profiling that has already been paid for elsewhere; the plans stay
    # this run's own either way, so a shared store never answers a point
    # with a plan another run searched.
    store = arguments.artifact_store or (base / "artifact_store")
    build_store = arguments.build_store
    plan_store = arguments.plan_store or (base / "plan_store")
    if suffix is not None:
        store = store / suffix
        build_store = None if build_store is None else build_store / suffix
        plan_store = plan_store / suffix
    record = _request_record(arguments, store, build_store, plan_store)
    if rank is not None:
        record["rank"] = rank

    # Everything printed is also kept beside the run it describes. The
    # progress log records what the search and the runtime were doing at each
    # moment; this is the report a person actually read -- the geometry table,
    # the chosen plans, the per-step numbers -- and a run whose console has
    # scrolled away is a measurement that has to be taken again to be read.
    run_root.mkdir(parents=True, exist_ok=True)
    # The request in full, so a later run can repeat it without the command.
    (run_root / "request.json").write_text(
        json.dumps(
            record,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return RunPaths(run_root, store, build_store, plan_store)


@contextmanager
def console_log(root: Path) -> Iterator[None]:
    """Mirror stdout/stderr for this invocation, restoring them on every exit."""
    with (root / "console.log").open("w", encoding="utf-8") as console:

        class Tee:
            def __init__(self, terminal: TextIO) -> None:
                self.terminal = terminal

            def write(self, text: str) -> int:
                console.write(text)
                console.flush()
                return self.terminal.write(text)

            def flush(self) -> None:
                console.flush()
                self.terminal.flush()

            def __getattr__(self, name: str) -> object:
                return getattr(self.terminal, name)

        with redirect_stdout(Tee(sys.stdout)), redirect_stderr(Tee(sys.stderr)):
            yield


class Ledger(dict[str, float]):
    """Where the wall clock went, by category, for the closing table."""

    def __init__(self) -> None:
        super().__init__()
        self.started = time.perf_counter()

    def charge(self, category: str, started: float) -> None:
        self[category] = self.get(category, 0.0) + (time.perf_counter() - started)
