"""One line of progress per phase, and a stack dump if a canary wedges.

The canaries report by exit status and otherwise stay quiet, which is right
until one exceeds its CTest timeout: then the log holds nothing, and a
120-second silence could be any of six phases. A marker on stderr at each
phase boundary costs nothing and turns the next timeout into a location.

A location is not always enough. A canary that wedges inside the runtime
leaves no clue about *which* wait it is in, and CTest's timeout kills it
without one. So a watchdog is armed here too: shortly before CTest would
give up, every thread's Python stack is printed and the process exits. The
difference between a silent kill and a printed stack is the difference
between a day of bisecting and a minute of reading, and it costs one timer.

The watchdog fires at ``SHADOWSPILL_CANARY_TIMEOUT`` seconds minus a margin;
CMake sets that from the same value it gives CTest, so the two never drift.
Without it the watchdog is off, which is what a canary run by hand wants.
"""

from __future__ import annotations

import faulthandler
import os
import sys
import time

#: Fire this far before CTest would kill us, so the dump reaches the log.
_WATCHDOG_MARGIN_SECONDS = 15.0
#: Below this there is no room for a dump before the kill, so do not arm.
_WATCHDOG_MINIMUM_SECONDS = 5.0

_started = time.monotonic()
_current = "start"


def phase(name: str) -> None:
    """Announce the phase a canary is entering, with seconds since it began."""
    global _current
    _current = name
    print(
        f"phase {name} at {time.monotonic() - _started:.1f}s",
        file=sys.stderr,
        flush=True,
    )


def _arm_watchdog() -> None:
    """Dump every thread and exit if this canary outlives its budget."""

    budget = os.environ.get("SHADOWSPILL_CANARY_TIMEOUT")
    if not budget:
        return
    try:
        seconds = float(budget) - _WATCHDOG_MARGIN_SECONDS
    except ValueError:
        return
    if seconds < _WATCHDOG_MINIMUM_SECONDS:
        return
    # Faults print a stack too, which is worth having whatever the cause.
    faulthandler.enable()
    print(
        f"canary watchdog armed for {seconds:.0f}s",
        file=sys.stderr,
        flush=True,
    )
    faulthandler.dump_traceback_later(seconds, exit=True)


_arm_watchdog()
