"""Physical-budget reconciliation and exact slab replay."""

from shadowspill.planner.admission.admission_replay import (
    AdmissionReplay,
    AdmissionReplayPurpose,
    AdmissionReplayStep,
    CausalAdmissionDependency,
    OwnershipTransition,
    OwnershipTransitionKind,
)
from shadowspill.planner.admission.layout import (
    FixedLayoutAdmission,
    FixedLayoutInfeasibleError,
    FixedLayoutMeasurement,
    FixedLayoutPlacement,
    FixedLayoutReuse,
    FixedPhysicalLayout,
    build_fixed_layout_admission,
    certify_fixed_layout,
    measure_fixed_layout,
)
from shadowspill.planner.admission.refinement import (
    FixedLayoutAttempt,
    FixedLayoutSelection,
    placement_facts,
    resolve_fixed_layout_selection,
)
from shadowspill.planner.admission.simulation import simulation_admission_from_replay

from .bindings import TaskOutputBinding, build_admission_facts
from .layout_runtime import (
    DynamicTaskAllocationPolicy,
    project_runtime_fixed_layout,
)
from .physical import physical_admission, reconcile_spill_pool, seal_physical_budget
from .selection import (
    SelectedAdmission,
    build_fixed_selected_admission,
    dynamic_scratch_reserve_bytes,
    output_bindings_for_entrypoints,
)

__all__ = [
    "AdmissionReplay",
    "AdmissionReplayPurpose",
    "AdmissionReplayStep",
    "CausalAdmissionDependency",
    "DynamicTaskAllocationPolicy",
    "FixedLayoutAdmission",
    "FixedLayoutAttempt",
    "FixedLayoutInfeasibleError",
    "FixedLayoutMeasurement",
    "FixedLayoutPlacement",
    "FixedLayoutReuse",
    "FixedLayoutSelection",
    "FixedPhysicalLayout",
    "OwnershipTransition",
    "OwnershipTransitionKind",
    "SelectedAdmission",
    "TaskOutputBinding",
    "build_admission_facts",
    "build_fixed_layout_admission",
    "build_fixed_selected_admission",
    "certify_fixed_layout",
    "dynamic_scratch_reserve_bytes",
    "measure_fixed_layout",
    "output_bindings_for_entrypoints",
    "physical_admission",
    "placement_facts",
    "project_runtime_fixed_layout",
    "reconcile_spill_pool",
    "resolve_fixed_layout_selection",
    "seal_physical_budget",
    "simulation_admission_from_replay",
]
