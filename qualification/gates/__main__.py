"""Run suite, numerical and performance gates in order, without GPU overlap.

SSD and remote gates are opt-in; each requires its corresponding spill backend.
Per-gate arguments belong in a JSON configuration rather than this wrapper.
"""

import argparse
import sys
from pathlib import Path

from .config import ALL_GATES, GATE_ORDER, gate_options
from .environment import _default_run, _refuse_a_stale_library
from .report import _summary
from .run import run_gates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "gates",
        nargs="*",
        choices=ALL_GATES,
        default=None,
        help="gates to run in fixed order; default: suite, numerical, performance",
    )
    parser.add_argument(
        "--run",
        help="result directory suffix; default: revision and timestamp",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="let a matrix finish its remaining cells after one fails",
    )
    parser.add_argument(
        "--continue-after-failure",
        action="store_true",
        help="run later gates even after one fails",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON object mapping gate names to lists of command-line arguments",
    )
    parser.add_argument(
        "--allow-stale-library",
        action="store_true",
        help="deliberately measure a build older than its C sources",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        help="CPU threads per process; default: all CPUs available by affinity",
    )
    arguments = parser.parse_args()
    if arguments.cpu_threads is not None and arguments.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    if not arguments.allow_stale_library:
        stale = _refuse_a_stale_library()
        if stale is not None:
            print(f"refusing to run the gates: {stale}", file=sys.stderr)
            return 2

    run = arguments.run or _default_run()
    outcomes = run_gates(
        arguments.gates or list(GATE_ORDER),
        run=run,
        keep_going=arguments.keep_going,
        continue_after_failure=arguments.continue_after_failure,
        options=gate_options(arguments.config),
        cpu_threads=arguments.cpu_threads,
    )
    print()
    print(_summary(f"gates {run}", outcomes, run))
    return 0 if all(outcome.passed for outcome in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
