"""One task's shape and provenance, framework-free.

`shadowspill.step` is a training step's; this is one task's. A frontend captures
a task, compiles it and measures it, and what it produces is these records: the
storage contract the task promised, the allocations it made and the path they
took, and the measurement itself. Everything downstream -- planning, simulation,
the runtime, the diagnostics -- reads them and no framework.

`storage` is what a task promised about its storages: its roots, the views on its
outputs, and the mutations it declares. `allocations` is what it promised about
its allocations, and what the profiled run observed. `profiles` is one
measurement and the key it is filed under. `inputs` is the vocabulary for a task
argument and the provenance of the value profiled for it.
"""

from .inputs import (
    REPRESENTATIVE_VALUE_POLICY,
    RepresentativeInputSummary,
    TaskInputRole,
)

__all__ = [
    "REPRESENTATIVE_VALUE_POLICY",
    "RepresentativeInputSummary",
    "TaskInputRole",
]
