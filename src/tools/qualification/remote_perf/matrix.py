"""Full-model throughput with the spill pool on another machine.

**This is the performance gate's matrix, with one substitution.** Same cells,
same manifests, same protocol -- same groups, same steps per group, same
budgets; the spill pool is a region a daemon holds on a peer instead of pinned
host memory. Keeping everything else identical is what makes the two runs
comparable, and comparability is the entire point.

It is not in the default gate run. It needs a memory daemon reachable at
``SHADOWSPILL_NETWORK_PEER`` (``host:port``) and **skips cleanly without one**,
writing a summary that says so rather than failing.

Its cells are judged against floors of their own,
``remote_regression_tokens_per_second`` on the manifests, measured with the pool
on a peer; the local floors do not apply, because the interconnect is about
3 GB/s against 25 GB/s to pinned host memory and a transfer-bound cell runs
several times slower. The local matrix cannot tell whether a change helped or
hurt a run that spills across a network; this one can.

The peer's pool defaults to the manifests' own spill budget and can be made
smaller with ``--remote-spill-gib``, which matters because that budget is
112 GiB and a peer may not have it. Anything smaller is a second thing
differing from the local run -- planning sees a smaller pool and may choose
differently -- so the flag says what it costs rather than being silent.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from shadowspill.network import remote

#: What the full-model manifests ask for. Named here so a smaller pool is
#: visibly a departure rather than a different default.
DEFAULT_SPILL_GIB = 112


def peer_from_environment() -> tuple[str, int] | None:
    """Where the daemon is, or ``None`` when nobody said."""

    configured = os.environ.get("SHADOWSPILL_NETWORK_PEER", "").strip()
    if not configured:
        return None
    host, separator, port = configured.rpartition(":")
    if not separator or not host or not port.isdigit():
        raise SystemExit(
            f"SHADOWSPILL_NETWORK_PEER must read host:port, not {configured!r}"
        )
    return host, int(port)


def _skip(output_directory: Path, reason: str) -> int:
    """Record a skip the gate summary can read, and succeed.

    Written as a summary rather than printed and forgotten, because the gate's
    report reads the same file for a run and for a skip, and a gate that says
    nothing is indistinguishable from one that was never run.
    """

    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "summary.json").write_text(
        json.dumps({"skipped": True, "reason": reason, "cells": []}, indent=2) + "\n"
    )
    print(f"remote_perf gate skipped: {reason}")
    return 0


def _spill_gib(arguments: list[str]) -> tuple[int, list[str]]:
    """Read and remove this layer's own option; the rest belong to the matrix."""

    remaining: list[str] = []
    capacity = DEFAULT_SPILL_GIB
    index = 0
    while index < len(arguments):
        if arguments[index] == "--remote-spill-gib" and index + 1 < len(arguments):
            value = arguments[index + 1]
            if not value.isdigit() or int(value) <= 0:
                raise SystemExit(
                    f"--remote-spill-gib must be a positive integer, not {value!r}"
                )
            capacity = int(value)
            index += 2
            continue
        remaining.append(arguments[index])
        index += 1
    return capacity, remaining


def main() -> int:
    # Parsed before anything expensive: the arguments belong to the performance
    # matrix, and the only ones this layer reads are its own.
    capacity_gib, arguments = _spill_gib(sys.argv[1:])
    sys.argv = [sys.argv[0], *arguments]

    output_directory = Path("qualification/results/remote_perf")
    for index, argument in enumerate(arguments):
        if argument == "--output-directory" and index + 1 < len(arguments):
            output_directory = Path(arguments[index + 1])

    peer = peer_from_environment()
    if peer is None:
        return _skip(
            output_directory,
            "SHADOWSPILL_NETWORK_PEER names no daemon",
        )
    host, port = peer

    # Imported here rather than at module scope: the performance matrix pulls in
    # the frontend and a framework, and a skip should not pay for that.
    from tools.qualification import performance_matrix

    # Every cell runs, even after one misses, so a run records all three
    # numbers rather than one. `--keep-going` is the matrix's own flag; naming
    # it here rather than requiring it of the caller is what makes the gate
    # usable.
    if "--keep-going" not in arguments:
        arguments = [*arguments, "--keep-going"]
        sys.argv = [sys.argv[0], *arguments]

    spill = remote(capacity=capacity_gib << 30, host=host, port=port)
    note = "" if capacity_gib == DEFAULT_SPILL_GIB else " (smaller than the manifests')"
    print(f"remote_perf gate: spilling to {host}:{port}, {capacity_gib} GiB{note}")
    # No cell list of its own, for the same reason the remote numerical gate has
    # none: the claim is that nothing differs but where the pool lives, so it
    # runs that matrix's cells by not choosing. `--cells` still narrows by hand.
    return performance_matrix.main_with_spill(spill)


if __name__ == "__main__":
    raise SystemExit(main())
