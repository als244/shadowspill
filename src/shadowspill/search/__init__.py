"""A step planned at every geometry, budget and walk, and the report that answers.

Framework-neutral: enumerating the splits of a sequence total, planning each
point against a saved program, and reading the answers back need no framework.
Building the programs a point is planned from does, so
the frontend's step search drives this package.

`geometries` enumerates the splits and the walks over each; `planner` answers one
point; `refusals` says why a point was refused rather than raising; `report` is
what a whole search answered, and reads one back.
"""

from .geometries import default_orderings, search_geometries
from .report import StepSearchGeometryBuild, StepSearchPoint, StepSearchReport

__all__ = [
    "StepSearchGeometryBuild",
    "StepSearchPoint",
    "StepSearchReport",
    "default_orderings",
    "search_geometries",
]
