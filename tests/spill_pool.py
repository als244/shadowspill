"""Where these tests spill, and how to put that somewhere unreadable.

A test written about what a plan *does* has no opinion about what kind of
memory it spills into, so the kind is a variable here rather than a literal at
each site. ``SHADOWSPILL_TEST_SPILL_PEER``, as ``host:port``, runs every one of
them against a pool whose memory is on another machine; unset, they spill to
pinned host memory exactly as before.

It exists because a frontend that assumes a pool address can be dereferenced
fails silently until something reads one. Four such assumptions were found by
walking the frontend and two more by a segfault in a gate -- and the two the
walk missed were missed because they never look a pool up at all, so no audit
of pool-consulting code could have reached them. Running the ordinary tests
against a pool that faults on a read is what turns the next one into a test
failure instead of a gate three weeks later.

Tests whose subject *is* the kind -- topology validation, the route rules, the
lookup accepting a kind the neutral configuration has never heard of -- name
their pool directly and do not come through here. For those, naming the kind is
the point rather than an incidental choice.
"""

from __future__ import annotations

import os
from typing import Any

from shadowspill.memory import pinned_host


def spill_pool(capacity: int) -> Any:
    """A spill pool of `capacity` bytes, on this machine or on a peer."""

    peer = os.environ.get("SHADOWSPILL_TEST_SPILL_PEER", "").strip()
    if not peer:
        return pinned_host(capacity=capacity)
    host, separator, port = peer.rpartition(":")
    if not separator or not host or not port.isdigit():
        raise ValueError(
            f"SHADOWSPILL_TEST_SPILL_PEER must read host:port, not {peer!r}"
        )
    # Imported here rather than at module scope: a run that names no peer
    # should not load the network package at all.
    from shadowspill.network import remote

    return remote(capacity=capacity, host=host, port=int(port))
