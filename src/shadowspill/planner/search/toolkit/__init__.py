"""What a search may call into, and what it need not write itself.

Everything here is search-agnostic: it takes a program, a machine, or a
plan, and says something true about it whichever search asked. A search
uses what it wants and ignores the rest -- nothing here is required, and
nothing here knows which search is running.

``validation``   whether a search was handed something it can work on
``resolution``   a program and a set of shares, as resolved programs
``alternatives``  what each alternative group offers, and what each costs

Two more toolkits sit outside this package because they are phases rather
than helpers: `shadowspill.simulator` prices a schedule, and
`shadowspill.planner.admission` proves one fits real memory. A search calls
those the same way.

This is public surface. A search written outside ShadowSpill imports from
here; see ``docs/architecture/search-algorithm.md``.
"""

from __future__ import annotations

from .resolution import (
    DEFAULT_RESOLUTION_OPTIONS,
    CostedAlternatives,
    Resolution,
    ShareValue,
    resolutions,
    validate_resolution_options,
)
from .validation import validate_search_inputs

__all__ = [
    "DEFAULT_RESOLUTION_OPTIONS",
    "CostedAlternatives",
    "Resolution",
    "ShareValue",
    "resolutions",
    "validate_resolution_options",
    "validate_search_inputs",
]
