"""Numerical correctness with the spill pool on another machine.

**This is the numerical gate's matrix, with one substitution.** Same programs,
same references, same tolerances, same routes; the spill pool is a region a
daemon holds on a peer instead of pinned host memory. That is the variable
under test, and keeping everything else identical is what makes a pass mean
something -- a disagreement can only be the pool.

It is not in the default gate run. It needs a memory daemon reachable at
``SHADOWSPILL_NETWORK_PEER`` (``host:port``) and **skips cleanly without one**,
writing a summary that says so rather than failing: a machine with no peer is
not a machine with a broken runtime.

There is deliberately **no performance cell**. The link is 25 Gb/s against
25.5 GB/s locally, so a rate measured over it is not comparable with anything
and is not a baseline. What this gate answers is whether the numbers come out
the same, which is a question the hardware can answer honestly.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from shadowspill.network import remote

#: One cell by default, the smallest. The matrix is otherwise unchanged, so
#: widening this is a list entry -- but every cell moves its whole spill volume
#: over a link 8x slower than local memory, and the gate is meant to be run
#: deliberately rather than on every change.
DEFAULT_CELLS = ("olmoe",)

#: What the peer is asked for. The numerical gate's spill budget is far larger
#: than a daemon on a shared box should claim, and the cells that run here are
#: chosen to fit it.
DEFAULT_SPILL_BYTES = 48 << 30


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

    A skip is written as a summary rather than printed and forgotten, because
    the gate's report reads the same file for a run and for a skip, and a gate
    that says nothing is indistinguishable from one that was never run.
    """

    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "summary.json").write_text(
        json.dumps({"skipped": True, "reason": reason, "cases": []}, indent=2) + "\n"
    )
    print(f"remote gate skipped: {reason}")
    return 0


def main() -> int:
    # Parsed before anything expensive: the arguments belong to the numerical
    # matrix, and the only ones this layer reads are its own.
    arguments = sys.argv[1:]
    output_directory = Path("qualification/results/remote")
    for index, argument in enumerate(arguments):
        if argument == "--output-dir" and index + 1 < len(arguments):
            output_directory = Path(arguments[index + 1])

    peer = peer_from_environment()
    if peer is None:
        return _skip(
            output_directory,
            "SHADOWSPILL_NETWORK_PEER names no daemon",
        )
    host, port = peer

    # Imported here rather than at module scope: the numerical matrix pulls in
    # the frontend and a framework, and a skip should not pay for that.
    from tools.qualification.numerical import matrix as numerical

    spill = remote(capacity=DEFAULT_SPILL_BYTES, host=host, port=port)
    print(
        f"remote gate: spilling to {host}:{port}, "
        f"{DEFAULT_SPILL_BYTES >> 30} GiB, cells {', '.join(DEFAULT_CELLS)}"
    )
    return numerical.main_with_spill(spill, default_models=DEFAULT_CELLS)


if __name__ == "__main__":
    raise SystemExit(main())
