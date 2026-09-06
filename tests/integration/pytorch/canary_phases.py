"""One line of progress per phase, so a canary that times out says where.

The canaries report by exit status and otherwise stay quiet, which is right
until one exceeds its CTest timeout: then the log holds nothing, and a
120-second silence could be any of six phases. A marker on stderr at each
phase boundary costs nothing and turns the next timeout into a location.
"""

from __future__ import annotations

import sys
import time

_started = time.monotonic()


def phase(name: str) -> None:
    """Announce the phase a canary is entering, with seconds since it began."""
    print(
        f"phase {name} at {time.monotonic() - _started:.1f}s",
        file=sys.stderr,
        flush=True,
    )
