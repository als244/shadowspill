"""Pool occupancy and timelines for stored plans and traced steps.

Calculations, rendering, stored-run discovery and the CLI are separate modules.
The functions re-exported here retain the existing API.
"""

from .cli import main
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
from .run.records import SUMMARY_COLUMNS
from .run.write import recompute_share_of, write_run_timelines
from .spill import spill_occupancy
from .stored import StoredPlan, program_path_for, stored_plans
from .summary import summarize

__all__ = [
    "CATEGORY_ORDER",
    "ROLE_PRIORITY",
    "SUMMARY_COLUMNS",
    "UNATTRIBUTED",
    "WORKSPACE",
    "Clock",
    "Interval",
    "LeaseClock",
    "ObjectFacts",
    "Occupancy",
    "PlanOccupancy",
    "ProgramFacts",
    "StoredPlan",
    "TaskSpan",
    "TransferSpan",
    "all_save",
    "attribute",
    "categorize",
    "cheapest_selections",
    "describe_plan",
    "execution_occupancy",
    "main",
    "page_data",
    "placement_rows",
    "program_path_for",
    "recompute_share_of",
    "selected_task_ids",
    "spill_occupancy",
    "storage_handoffs",
    "stored_plans",
    "summarize",
    "table",
    "task_spans",
    "unconstrained",
    "write_pages",
    "write_run_timelines",
]
