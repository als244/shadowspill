"""Gate order, opaque per-gate options, and subprocess commands."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from qualification.pipeline.process import python_command

GATE_ORDER = ("suite", "numerical", "performance")
ALL_GATES = (
    "suite",
    "numerical",
    "numerical_ssd",
    "performance",
    "remote",
    "remote_perf",
)
RESULTS = Path("qualification/results")


def gate_options(path: Path | None) -> dict[str, tuple[str, ...]]:
    """Read one config into the arguments each gate should be given."""

    empty: dict[str, tuple[str, ...]] = {name: () for name in ALL_GATES}
    if path is None:
        return empty
    try:
        loaded = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise SystemExit(f"{path}: not valid JSON: {error}") from error
    if not isinstance(loaded, dict):
        raise SystemExit(
            f"{path}: expected an object with a section per gate, "
            f"found {type(loaded).__name__}"
        )
    unknown = sorted(set(loaded) - set(ALL_GATES))
    if unknown:
        raise SystemExit(
            f"{path}: unknown gate section {unknown}; sections are {list(ALL_GATES)}"
        )
    options = dict(empty)
    for name, values in loaded.items():
        if not isinstance(values, list) or not all(
            isinstance(value, str) for value in values
        ):
            raise SystemExit(
                f"{path}: section {name!r} must be a list of command-line "
                "arguments, as strings"
            )
        options[name] = tuple(values)
    return options


def _commands(
    name: str,
    run: str,
    keep_going: bool,
    *,
    options: Sequence[str] = (),
) -> tuple[str, ...]:
    """The command one gate runs, named for the run it belongs to."""

    if name == "suite":
        # Pyproject supplies -q; adding it again suppresses summary counts.
        return (
            *python_command("pytest"),
            "tests",
            "-p",
            "no:cacheprovider",
            *options,
        )
    matrices = {
        "numerical": ("qualification.numerical.matrix", "--output-dir"),
        "numerical_ssd": ("qualification.numerical.matrix", "--output-dir"),
        "performance": ("qualification.performance.matrix", "--output-directory"),
        "remote": ("qualification.remote.matrix", "--output-dir"),
        "remote_perf": (
            "qualification.remote_perf.matrix",
            "--output-directory",
        ),
    }
    if name not in matrices:
        raise KeyError(f"no command is defined for the gate {name!r}")
    module, output_flag = matrices[name]
    command = [
        *python_command(module),
        output_flag,
        str(RESULTS / f"{name}_{run}"),
    ]
    if keep_going:
        command.append("--keep-going")
    # Fresh compiler caches, retained phase artifacts.
    command.append("--empty-caches")
    command += options
    if name == "numerical_ssd":
        command += ["--spill-pool", "ssd"]
    return tuple(command)
