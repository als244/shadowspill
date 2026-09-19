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

This gate carries **no performance cell**, and that is a division of labour
rather than a judgement: what it answers is whether the numbers come out the
same, which is a question the hardware can answer honestly whatever the link
costs. Throughput over that link is ``remote_perf``'s question.

An earlier version of this note argued no such gate could exist, on the grounds
that a rate measured over a 25 Gb/s link is not comparable with anything. That
was half right. It is not comparable with the **local** matrix -- and it does
not need to be. It is comparable with itself over time, which is what a
baseline is, and without one nothing can say whether a change helped a run that
spills across a network.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from shadowspill.network import remote


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
    from tools.qualification.numerical.tolerances import SPILL_BUDGET

    # The peer is asked for the numerical gate's own spill budget, named once
    # where that gate defines it. Anything larger is refused by the gate's
    # verdict, which requires the pool to fit the budget as well as the peak to
    # fit the pool -- and asking for a different size would be one more thing
    # differing from the run this is compared against.
    spill = remote(capacity=SPILL_BUDGET, host=host, port=port)
    print(f"remote gate: spilling to {host}:{port}, {SPILL_BUDGET >> 30} GiB")
    # No cell list of its own. Naming one here would be a second matrix to keep
    # in step, and the whole claim this gate makes is that nothing differs from
    # the numerical run but where the spill pool lives -- so it runs that
    # matrix's cells by not choosing. `--models` still narrows a run by hand.
    return numerical.main_with_spill(spill)


if __name__ == "__main__":
    raise SystemExit(main())
