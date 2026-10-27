"""Run gates serially, streaming each process to the terminal and a log."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .config import ALL_GATES, RESULTS, _commands
from .environment import _cpu_environment, _host_state

_BANNER_WIDTH = 78


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


def _banner(text: str) -> str:
    """A rule that survives being read in a stream of other programs' output."""

    padded = f" {text} "
    stars = max(_BANNER_WIDTH - len(padded), 8)
    left = stars // 2
    return f"{'*' * left}{padded}{'*' * (stars - left)}"


def _stream(
    command: Sequence[str],
    log: Path,
    *,
    preamble: str = "",
    environment: Mapping[str, str] | None = None,
) -> int:
    """Run a gate, sending every line to this stdout and to its own log."""

    with log.open("w") as handle:
        if preamble:
            handle.write(preamble + "\n")
            handle.flush()
        process = subprocess.Popen(
            list(command),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert process.stdout is not None
        descriptor = process.stdout.fileno()
        while True:
            chunk = os.read(descriptor, 65536)
            if not chunk:
                break
            decoded = chunk.decode("utf-8", errors="replace")
            sys.stdout.write(decoded)
            sys.stdout.flush()
            handle.write(decoded)
            handle.flush()
        return process.wait()


def run_gates(
    gates: Sequence[str],
    *,
    run: str,
    keep_going: bool = False,
    continue_after_failure: bool = False,
    options: Mapping[str, Sequence[str]] | None = None,
    cpu_threads: int | None = None,
) -> list[GateOutcome]:
    """Run each gate in `GATE_ORDER`, newest output under `qualification/results`."""

    cpu_threads, environment = _cpu_environment(cpu_threads)
    ordered = [name for name in ALL_GATES if name in gates]
    logs = RESULTS / f"gates_{run}"
    logs.mkdir(parents=True, exist_ok=True)
    outcomes: list[GateOutcome] = []
    for name in ordered:
        command = _commands(
            name, run, keep_going, options=(options or {}).get(name, ())
        )
        log = logs / f"{name}.log"
        print(f"\n\n{_banner(f'START OF {name.upper()} GATE')}\n", flush=True)
        print(f"[{time.strftime('%H:%M:%S')}] {name}: {' '.join(command)}", flush=True)
        host = _host_state() + f"; CPU threads {cpu_threads}"
        print(host, flush=True)
        started = time.perf_counter()
        returncode = _stream(command, log, preamble=host, environment=environment)
        if returncode != 0:
            host = _host_state()
            print(host, flush=True)
            with log.open("a") as handle:
                handle.write(host + "\n")
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
