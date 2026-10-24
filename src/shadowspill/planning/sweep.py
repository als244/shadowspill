"""An optional loop over supplied Programs and budgets; no build or profiling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from shadowspill.artifacts.profiled_program import ProfiledProgram
from shadowspill.artifacts.sweep import SweepCase, SweepResult
from shadowspill.communication import CommunicationInfo
from shadowspill.errors import PlanInfeasibleError, PlanSearchExhaustedError

from .budgets import MemoryBudgets
from .plan import plan_program

if TYPE_CHECKING:
    from shadowspill.planner.search import SearchOptions


def sweep_programs(
    programs: Mapping[str, ProfiledProgram],
    *,
    memory_budgets: Sequence[MemoryBudgets],
    communication_info: CommunicationInfo,
    search_options: SearchOptions | None = None,
) -> SweepResult:
    """Keep each result. Invalid inputs and implementation errors are not hidden."""
    cases = []
    for budget in memory_budgets:
        for name, program in programs.items():
            try:
                plan = plan_program(
                    program,
                    memory_budgets=budget,
                    communication_info=communication_info,
                    search_options=search_options,
                )
            except (PlanInfeasibleError, PlanSearchExhaustedError) as error:
                failure: dict[str, Any] = {
                    "type": type(error).__name__,
                    "message": str(error),
                }
                for key in (
                    "kind",
                    "device_id",
                    "boundary_task_id",
                    "required_bytes",
                    "capacity_bytes",
                ):
                    if hasattr(error, key):
                        failure[key] = getattr(error, key)
                failure["diagnostics"] = [
                    item.to_dict() if hasattr(item, "to_dict") else str(item)
                    for item in error.diagnostics
                ]
                cases.append(SweepCase(name, budget, failure=failure))
            else:
                cases.append(SweepCase(name, budget, plan))
    return SweepResult(tuple(cases))
