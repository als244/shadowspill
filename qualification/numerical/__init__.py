"""Fresh-process compiled-reference/planned numerical qualification.

One case is two arms of the same step: the reference, fully compiled without
ShadowSpill, and the planned run. Each arm runs in its own process; the
planned arm is then compared against the reference and against its own
checkpoint replay, and everything either comparison used is written beside
the verdict.
"""

from __future__ import annotations

from .request import CaseRequest, PlannedRequest

__all__ = ["CaseRequest", "PlannedRequest"]
