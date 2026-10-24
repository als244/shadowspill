"""Transitional imports while existing runtime callers migrate to phase APIs."""

from shadowspill.planning.admission.facts import build_admission_facts
from shadowspill.planning.admission.outputs import (
    TaskOutputBinding,
    output_bindings_for_entrypoints,
)

__all__ = [
    "TaskOutputBinding",
    "build_admission_facts",
    "output_bindings_for_entrypoints",
]
