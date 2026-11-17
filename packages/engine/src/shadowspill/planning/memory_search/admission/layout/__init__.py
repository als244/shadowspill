"""Dependency-certified fixed physical layout admission."""

from shadowspill.planning.memory_search.admission.layout.build import (
    FixedLayoutMeasurement,
    build_fixed_layout_admission,
    certify_fixed_layout,
    measure_fixed_layout,
)
from shadowspill.planning.memory_search.admission.layout.model import (
    FixedLayoutAdmission,
    FixedLayoutInfeasibleError,
    FixedLayoutPlacement,
    FixedLayoutReuse,
    FixedPhysicalLayout,
)

__all__ = [
    "FixedLayoutAdmission",
    "FixedLayoutInfeasibleError",
    "FixedLayoutMeasurement",
    "FixedLayoutPlacement",
    "FixedLayoutReuse",
    "FixedPhysicalLayout",
    "build_fixed_layout_admission",
    "certify_fixed_layout",
    "measure_fixed_layout",
]
