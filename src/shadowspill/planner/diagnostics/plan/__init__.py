"""Immutable diagnostics describing one completed planning call.

One module per part of the record: what building it cost (``compilation``),
the graphs it captured (``graphs``), the task stages (``stages``), where the
leases went (``layout``), what the search did (``diagnostics``), the few
numbers that describe the plan (``summary``), and the whole record
(``report``).
"""

from __future__ import annotations

from .compilation import (
    PlanCacheArtifact,
    PlanCompilerProfile,
    PlanPhaseTiming,
    PlanProfilingMetadata,
)
from .diagnostics import PlanDiagnostics
from .graphs import (
    PlanAllocationABIStep,
    PlanAllocationEvent,
    PlanCompiledOutputView,
    PlanCompiledRoot,
    PlanGraphPair,
    PlanGraphProfile,
    PlanMutationBinding,
    PlanObjectFootprint,
    PlanOutputView,
    PlanRepresentativeInput,
    PlanStorageRoot,
    PlanUniqueStage,
)
from .layout import PlanFixedLayoutAttempt, PlanPhysicalLayout
from .report import PlanReport
from .stages import PlanTaskMemoryEnvelope, PlanTaskStage
from .summary import PlanSummary, summarize_selected_plan

__all__ = [
    "PlanAllocationABIStep",
    "PlanAllocationEvent",
    "PlanCacheArtifact",
    "PlanCompiledOutputView",
    "PlanCompiledRoot",
    "PlanCompilerProfile",
    "PlanDiagnostics",
    "PlanFixedLayoutAttempt",
    "PlanGraphPair",
    "PlanGraphProfile",
    "PlanMutationBinding",
    "PlanObjectFootprint",
    "PlanOutputView",
    "PlanPhaseTiming",
    "PlanPhysicalLayout",
    "PlanProfilingMetadata",
    "PlanReport",
    "PlanRepresentativeInput",
    "PlanStorageRoot",
    "PlanSummary",
    "PlanTaskMemoryEnvelope",
    "PlanTaskStage",
    "PlanUniqueStage",
    "summarize_selected_plan",
]
