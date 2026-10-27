"""Present experiment configuration and progress."""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path

from baselines.common.launch import engine_module
from baselines.sweep import reporting as sweep_runner
from baselines.sweep.config import expand

from .config import matrix_source, sweep_matrix, variable_values


class Console:
    """The experiment's console: each line printed and kept in the run's
    log."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.handle = path.open("a")

    def __call__(self, text: str = "") -> None:
        print(text, flush=True)
        self.handle.write(text + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def case_count(spec: dict, sweep: dict) -> str:
    """``25 cases``, from the sweep's matrix as the sweep expands it."""

    try:
        matrix = sweep_matrix(
            matrix_source(sweep), spec["settings"], variable_values(spec)
        )
        settings = engine_module(sweep["engine"]).Settings
        return f"{len(expand(matrix, spec['gpus'], settings))} cases"
    except (ValueError, ImportError) as error:
        return f"a matrix the sweep refuses ({error})"


def host_gib() -> float | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (ValueError, OSError, AttributeError):
        return None


def header(console: Console, spec: dict, spec_path: Path, selected: list, output: Path):
    """What the run is of: the spec, its settings, every sweep with its case
    count and whether this run runs it."""

    rule, field = sweep_runner.rule, sweep_runner.field
    console(rule(f"Experiment {Path(spec_path).resolve().parent.name}"))
    console(field("spec", spec_path))
    console(field("output", output))
    console(field("console", console.path))
    console(field("start", sweep_runner.clock()))
    memory = host_gib()
    console(
        field(
            "machine",
            f"{spec['gpus']} GPU{'s' if spec['gpus'] > 1 else ''}"
            + ("" if memory is None else f", {memory:.0f} GiB of host memory"),
        )
    )
    shared = " ".join(
        f"{key}={value if isinstance(value, str) else json.dumps(value)}"
        for key, value in spec["settings"].items()
    )
    for line in textwrap.wrap(
        shared,
        width=96,
        initial_indent=field("shared", ""),
        subsequent_indent=field("", ""),
        break_long_words=False,
        break_on_hyphens=False,
    ):
        console(line)
    width = max(len(sweep["name"]) for sweep in spec["sweeps"])
    for index, sweep in enumerate(spec["sweeps"]):
        role = "runs" if sweep in selected else "kept as it is"
        console(
            field(
                "sweeps" if index == 0 else "",
                f"{sweep['name']:<{width}}  {case_count(spec, sweep)} of"
                f" {sweep['engine']}; {role}",
            )
        )
