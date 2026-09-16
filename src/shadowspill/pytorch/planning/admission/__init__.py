"""The framework half of admission: entrypoints, bindings, and the selection.

A task entrypoint is a framework binding, so matching its output leaves to alias
groups and certifying the layout that follows are the frontend's. What a plan
needs of a live runtime is `shadowspill.pipeline.admission`.
"""

from shadowspill.pipeline.admission import SelectedAdmission
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

from .bindings import (
    TaskOutputBinding,
    build_admission_facts,
    output_bindings_for_entrypoints,
)
from .selection import build_fixed_selected_admission

__all__ = [
    "AdmissionReplay",
    "AdmissionReplayPurpose",
    "AdmissionReplayStep",
    "CausalAdmissionDependency",
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
    "measure_fixed_layout",
    "output_bindings_for_entrypoints",
    "placement_facts",
    "resolve_fixed_layout_selection",
    "simulation_admission_from_replay",
]
