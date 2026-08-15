"""Physical-budget reconciliation and exact slab replay."""

from .admission_replay import (
    AdmissionReplay,
    AdmissionReplayPurpose,
    AdmissionReplayStep,
    CausalAdmissionDependency,
    OwnershipTransition,
    OwnershipTransitionKind,
    replay_admission,
)
from .bindings import TaskOutputBinding, build_admission_topology
from .layout import (
    DynamicTaskAllocationPolicy,
    FixedLayoutAdmission,
    FixedLayoutInfeasibleError,
    FixedLayoutPlacement,
    FixedLayoutReuse,
    FixedPhysicalLayout,
    build_fixed_layout_admission,
    project_runtime_fixed_layout,
)
from .physical import physical_admission, reconcile_spill_pool, seal_physical_budget
from .refinement import (
    FixedLayoutAttempt,
    FixedLayoutSelection,
    resolve_fixed_layout_selection,
)
from .selection import (
    SelectedAdmission,
    admit_selected_schedule,
    build_fixed_selected_admission,
    build_selected_admission,
    output_bindings_for_entrypoints,
)
from .simulation import simulation_admission_from_replay

__all__ = [
    "AdmissionReplay",
    "AdmissionReplayPurpose",
    "AdmissionReplayStep",
    "CausalAdmissionDependency",
    "DynamicTaskAllocationPolicy",
    "FixedLayoutAdmission",
    "FixedLayoutAttempt",
    "FixedLayoutInfeasibleError",
    "FixedLayoutPlacement",
    "FixedLayoutReuse",
    "FixedLayoutSelection",
    "FixedPhysicalLayout",
    "OwnershipTransition",
    "OwnershipTransitionKind",
    "SelectedAdmission",
    "TaskOutputBinding",
    "admit_selected_schedule",
    "build_admission_topology",
    "build_fixed_layout_admission",
    "build_fixed_selected_admission",
    "build_selected_admission",
    "output_bindings_for_entrypoints",
    "physical_admission",
    "project_runtime_fixed_layout",
    "reconcile_spill_pool",
    "replay_admission",
    "resolve_fixed_layout_selection",
    "seal_physical_budget",
    "simulation_admission_from_replay",
]
