"""Run the validation gates in order, in one command.

The three gates answer different questions and are usually wanted together:
the unit suite says the tree is coherent, the numerical matrix says a planned
step still computes what the same step computes unplanned, and the
performance matrix says throughput has not regressed and the simulator still
predicts it. Running
them by hand means three commands, three output directories to name
consistently, and remembering the order.

Each gate runs to completion before the next begins. The GPU ones are timed
measurements, so overlapping them would corrupt both.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Gate names in the order they run. The suite is first because it is the
#: cheapest and catches what would make the measured gates meaningless.
GATE_ORDER = ("suite", "numerical", "performance")

_RESULTS = Path("qualification/results")


def gate_options(path: Path | None) -> dict[str, tuple[str, ...]]:
    """Read one config into the arguments each gate should be given.

    The file has a section per gate, each holding that gate's own command
    line. Keeping every gate in one file is what makes a run reproducible
    from a single artifact, and keeping the sections opaque is what stops
    this wrapper from having to mirror three other command lines.
    """

    empty: dict[str, tuple[str, ...]] = {name: () for name in GATE_ORDER}
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
    unknown = sorted(set(loaded) - set(GATE_ORDER))
    if unknown:
        raise SystemExit(
            f"{path}: unknown gate section {unknown}; sections are {list(GATE_ORDER)}"
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


@dataclass(frozen=True, slots=True)
class GateOutcome:
    """What one gate did."""

    name: str
    command: tuple[str, ...]
    returncode: int
    seconds: float
    log: Path

    @property
    def passed(self) -> bool:
        return self.returncode == 0


#: How wide a gate banner is drawn.
_BANNER_WIDTH = 78


def _banner(text: str) -> str:
    """A rule that survives being read in a stream of other programs' output."""

    padded = f" {text} "
    stars = max(_BANNER_WIDTH - len(padded), 8)
    left = stars // 2
    return f"{'*' * left}{padded}{'*' * (stars - left)}"


def _commands(
    name: str,
    run: str,
    keep_going: bool,
    *,
    options: Sequence[str] = (),
) -> tuple[str, ...]:
    """The command one gate runs, named for the run it belongs to.

    ``options`` is that gate's section of the config, forwarded verbatim. The
    wrapper deliberately does not know what any of them mean: every option it
    understood would be one it had to gain whenever a matrix gained one, and
    the two would drift.
    """

    if name == "suite":
        # No -q here: pyproject already passes one, and a second makes it
        # -qq, which drops the closing count the summary reads.
        return (
            sys.executable,
            "-u",
            "-m",
            "pytest",
            "tests",
            "-p",
            "no:cacheprovider",
            *options,
        )
    if name == "numerical":
        command = [
            sys.executable,
            "-u",
            "-m",
            "qualification.numerical.matrix",
            "--output-dir",
            str(_RESULTS / f"numerical_{run}"),
        ]
    else:
        command = [
            sys.executable,
            "-u",
            "-m",
            "qualification.performance.matrix",
            "--output-directory",
            str(_RESULTS / f"performance_{run}"),
        ]
    if keep_going:
        command.append("--keep-going")
    command += options
    return tuple(command)


#: How many failing tests the summary names before pointing at the log.
_NAMED_FAILURE_LIMIT = 15

#: pytest's closing line, whatever mix of outcomes it reports.
_PYTEST_TALLY = re.compile(r"(\d+) (passed|failed|skipped|deselected|errors?|xfailed)")


def _suite_report(log: Path) -> list[str]:
    """What the unit suite counted, from its own closing line."""

    try:
        text = log.read_text()
    except OSError:
        return ["    the suite wrote no readable log"]
    lines = text.splitlines()
    tallies: list[tuple[str, str]] = []
    for line in reversed(lines):
        tallies = _PYTEST_TALLY.findall(line)
        if tallies:
            break
    if not tallies:
        rows = ["    the suite reported no counts"]
    else:
        counts = {name: int(count) for count, name in tallies}
        # Not "deselected": the fresh-process tests it counts are checked by
        # CTest inside this same gate, so reporting them here would say
        # something went unchecked when nothing did.
        counts.pop("deselected", None)
        rows = ["    " + ", ".join(f"{count} {name}" for name, count in counts.items())]
    # pytest's own short summary already names every test that did not pass,
    # and a count alone sends the reader back to the log for the one thing
    # they need.
    named = [line.strip() for line in lines if line.startswith(("FAILED ", "ERROR "))]
    rows.extend(f"      {line}" for line in named[:_NAMED_FAILURE_LIMIT])
    if len(named) > _NAMED_FAILURE_LIMIT:
        rows.append(f"      ... and {len(named) - _NAMED_FAILURE_LIMIT} more, in {log}")
    return rows


def _numerical_report(directory: Path) -> list[str]:
    """Which correctness cells agreed with their reference."""

    summary = directory / "summary.json"
    try:
        cases = json.loads(summary.read_text()).get("cases", [])
    except (OSError, json.JSONDecodeError):
        return [f"    no readable summary at {summary}"]
    passed = [case for case in cases if case.get("passed")]
    rows = [f"    {len(passed)}/{len(cases)} cells passed"]
    rows.extend(
        f"      {case.get('implementation')}_{case.get('family')}: FAILED"
        for case in cases
        if not case.get("passed")
    )
    return rows


def _performance_report(directory: Path) -> list[str]:
    """Throughput and planning time per cell, measured against predicted."""

    summary = directory / "summary.json"
    try:
        cells = json.loads(summary.read_text()).get("cells", [])
    except (OSError, json.JSONDecodeError):
        return [f"    no readable summary at {summary}"]
    results: list[tuple[str, dict[str, Any]]] = []
    for cell in cells:
        artifact = Path(str(cell.get("artifact", "")))
        try:
            results.append(
                (str(cell.get("identity")), json.loads(artifact.read_text()))
            )
        except (OSError, json.JSONDecodeError):
            continue
    if not results:
        return ["    no cell artifacts to report"]

    passed = sum(1 for _, value in results if value.get("passed"))
    rows = [f"    {passed}/{len(results)} cells passed", ""]
    rows.append(
        f"    {'cell':<14} {'real s':>8} {'real tok/s':>11} {'sim s':>8} "
        f"{'sim tok/s':>10} {'sim err':>8} {'plan s':>8}"
    )
    for identity, value in results:
        tokens = float(value["manifest"]["tokens_per_step"])
        simulated = float(value["predicted_makespan_seconds"])
        rows.append(
            f"    {identity:<14} {float(value['median_step_seconds']):8.3f} "
            f"{float(value['median_tokens_per_second']):11.1f} {simulated:8.3f} "
            f"{tokens / simulated:10.1f} "
            f"{float(value['simulator_relative_error']) * 100:+7.2f}% "
            f"{float(value['planning_seconds']):8.1f}"
        )

    phases: dict[str, dict[str, float]] = {}
    for identity, value in results:
        for phase, seconds in value.get("phase_seconds", {}).items():
            if phase != "total":
                phases.setdefault(phase, {})[identity] = float(seconds)
    ranked = sorted(
        phases.items(), key=lambda item: -max(item[1].values(), default=0.0)
    )
    material = [(name, times) for name, times in ranked if max(times.values()) >= 1.0]
    if material:
        rows.extend(["", "    planning time by phase, seconds"])
        names = [identity for identity, _ in results]
        # Wide enough for the longest phase there actually is, so no row pushes
        # its own numbers out of the column they belong to.
        width = max(len(phase) for phase, _ in material) + 2
        rows.append(
            "    " + f"{'phase':<{width}}" + "".join(f"{name:>14}" for name in names)
        )
        for phase, times in material:
            rows.append(
                f"    {phase:<{width}}"
                + "".join(f"{times.get(name, 0.0):>14.1f}" for name in names)
            )
    return rows


def _default_run() -> str:
    """Name a run for the commit it measured.

    A cell's saved numbers do not record which revision produced them, so the
    directory name is what makes a result usable later as a reference. A tree
    with uncommitted changes says so in the name, because a revision alone
    does not identify the code that ran.
    """

    try:
        revision = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        modified = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return time.strftime("%m%d_%H%M")
    return f"{revision}{'_dirty' if modified else ''}_{time.strftime('%m%d_%H%M')}"


def _stream(command: Sequence[str], log: Path) -> int:
    """Run a gate, sending every line to this stdout and to its own log.

    A gate is long enough that watching it matters, and its log is what the
    summary reads afterwards, so the output goes to both rather than to one
    and then the other.

    Two things have to be right for that to actually stream. The child must
    not block-buffer its own output because its stdout is a pipe rather than
    a terminal, which is what ``-u`` on each gate command prevents; and this
    end must not wait for whole lines, which is what reading the descriptor
    directly avoids.
    """

    with log.open("w") as handle:
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert process.stdout is not None
        descriptor = process.stdout.fileno()
        while True:
            # Whatever has arrived, rather than whatever completes a line: a
            # gate that reports progress as characters without newlines --
            # pytest's dots -- would otherwise show nothing at all until it
            # finished, which is the case where watching matters most.
            chunk = os.read(descriptor, 65536)
            if not chunk:
                break
            decoded = chunk.decode("utf-8", errors="replace")
            sys.stdout.write(decoded)
            sys.stdout.flush()
            handle.write(decoded)
        return process.wait()


def _summary(line: str, outcomes: Sequence[GateOutcome], run: str) -> str:
    width = max(len(outcome.name) for outcome in outcomes)
    rows = [line]
    for outcome in outcomes:
        verdict = "PASS" if outcome.passed else f"FAIL ({outcome.returncode})"
        rows.append(
            f"  {outcome.name:<{width}}  {verdict:<10} "
            f"{outcome.seconds / 60:6.1f} min  {outcome.log}"
        )
        if outcome.name == "suite":
            rows.extend(_suite_report(outcome.log))
        elif outcome.name == "numerical":
            rows.extend(_numerical_report(_RESULTS / f"numerical_{run}"))
        else:
            rows.extend(_performance_report(_RESULTS / f"performance_{run}"))
    return "\n".join(rows)


def run_gates(
    gates: Sequence[str],
    *,
    run: str,
    keep_going: bool = False,
    continue_after_failure: bool = False,
    options: Mapping[str, Sequence[str]] | None = None,
) -> list[GateOutcome]:
    """Run each gate in `GATE_ORDER`, newest output under `qualification/results`."""

    ordered = [name for name in GATE_ORDER if name in gates]
    logs = _RESULTS / f"gates_{run}"
    logs.mkdir(parents=True, exist_ok=True)
    outcomes: list[GateOutcome] = []
    for name in ordered:
        command = _commands(
            name, run, keep_going, options=(options or {}).get(name, ())
        )
        log = logs / f"{name}.log"
        # Three gates stream into one terminal, and a matrix prints its own
        # banners, so without a rule of their own the boundary between two
        # gates is a line that looks like any other.
        print(f"\n\n{_banner(f'START OF {name.upper()} GATE')}\n", flush=True)
        print(f"[{time.strftime('%H:%M:%S')}] {name}: {' '.join(command)}", flush=True)
        started = time.perf_counter()
        returncode = _stream(command, log)
        outcome = GateOutcome(
            name=name,
            command=command,
            returncode=returncode,
            seconds=time.perf_counter() - started,
            log=log,
        )
        outcomes.append(outcome)
        verdict = "passed" if outcome.passed else f"FAILED ({outcome.returncode})"
        print(
            f"[{time.strftime('%H:%M:%S')}] {name}: {verdict} in "
            f"{outcome.seconds / 60:.1f} min, log {log}",
            flush=True,
        )
        print(
            f"\n{
                _banner(
                    f'END OF {name.upper()} GATE'
                    f' [{verdict.split()[0]}, {outcome.seconds / 60:.1f} min]'
                )
            }"
            "\n\n",
            flush=True,
        )
        if not outcome.passed and not continue_after_failure:
            print(f"stopping: {name} failed and --continue-after-failure is not set")
            break
    return outcomes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "gates",
        nargs="*",
        choices=GATE_ORDER,
        # Not a list default: argparse validates a default against `choices`
        # when nargs is "*" and nothing was given, and a list is not one of
        # them, so `qualification.gates` with no arguments would refuse to run.
        default=None,
        help=(
            "which gates to run, in any order on the command line; they always "
            "run suite, numerical, performance. Default: all three"
        ),
    )
    parser.add_argument(
        "--run",
        # Resolved after parsing, not while building the parser: naming a run
        # asks git for the revision, and --help should not.
        default=None,
        help=(
            "names this run's output: qualification/results/numerical_<run>, "
            "performance_<run>, and gates_<run>/ for the logs. Defaults to the "
            "commit being measured and the date, which is what makes a result "
            "readable as a reference later"
        ),
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="let a matrix finish its remaining cells after one fails",
    )
    parser.add_argument(
        "--continue-after-failure",
        action="store_true",
        help="run the later gates even after one fails",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "JSON file with a section per gate -- suite, numerical, "
            "performance -- each holding that gate's own command-line "
            "arguments, forwarded verbatim. Options belong here rather than "
            "on this wrapper, which would otherwise have to mirror every "
            "matrix's command line"
        ),
    )
    arguments = parser.parse_args()

    run = arguments.run or _default_run()
    outcomes = run_gates(
        arguments.gates or list(GATE_ORDER),
        run=run,
        keep_going=arguments.keep_going,
        continue_after_failure=arguments.continue_after_failure,
        options=gate_options(arguments.config),
    )
    print()
    print(_summary(f"gates {run}", outcomes, run))
    return 0 if all(outcome.passed for outcome in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
