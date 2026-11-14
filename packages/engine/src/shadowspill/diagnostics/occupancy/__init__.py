"""Pool occupancy and timelines for stored plans and traced steps.

Calculations, rendering and the per-plan CLI are separate modules.
"""

from .clock import Clock
from .execution import (
    LeaseClock,
    execution_occupancy,
    placement_rows,
    storage_handoffs,
    task_spans,
)
from .facts import ProgramFacts, categorize, selected_task_ids
from .intervals import (
    CATEGORY_ORDER,
    ROLE_PRIORITY,
    UNATTRIBUTED,
    WORKSPACE,
    Interval,
    ObjectFacts,
    Occupancy,
    TaskSpan,
    TransferSpan,
)
from .page import describe_plan, page_data, table, write_pages
from .plan import PlanOccupancy, all_save, attribute, cheapest_selections, unconstrained
from .spill import spill_occupancy
from .stored import program_path_for
from .summary import summarize

__all__ = [
    "CATEGORY_ORDER",
    "ROLE_PRIORITY",
    "UNATTRIBUTED",
    "WORKSPACE",
    "Clock",
    "Interval",
    "LeaseClock",
    "ObjectFacts",
    "Occupancy",
    "PlanOccupancy",
    "ProgramFacts",
    "TaskSpan",
    "TransferSpan",
    "all_save",
    "attribute",
    "categorize",
    "cheapest_selections",
    "describe_plan",
    "execution_occupancy",
    "page_data",
    "placement_rows",
    "program_path_for",
    "selected_task_ids",
    "spill_occupancy",
    "storage_handoffs",
    "summarize",
    "table",
    "task_spans",
    "unconstrained",
    "write_pages",
]
