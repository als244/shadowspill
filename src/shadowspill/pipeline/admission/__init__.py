"""What a plan needs of a live runtime, and whether it fits.

`layout_runtime` projects a certified fixed layout onto the runtime's own
records; `physical` reconciles the predicted peak against the pool and seals the
budget. Certifying the layout in the first place reads framework entrypoints, so
that half is the frontend's admission package.
"""

from .layout_runtime import (
    DynamicTaskAllocationPolicy,
    project_runtime_fixed_layout,
)
from .physical import physical_admission, reconcile_spill_pool, seal_physical_budget
from .selected import (
    SelectedAdmission,
    dynamic_scratch_reserve_bytes,
    task_memory_envelope,
)

__all__ = [
    "DynamicTaskAllocationPolicy",
    "SelectedAdmission",
    "dynamic_scratch_reserve_bytes",
    "physical_admission",
    "project_runtime_fixed_layout",
    "reconcile_spill_pool",
    "seal_physical_budget",
    "task_memory_envelope",
]
