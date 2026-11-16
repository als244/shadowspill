"""Use the detailed occupancy renderer with the public, saved Plan artifacts."""

from collections.abc import Mapping
from typing import Any

from shadowspill.artifacts.plan import Plan
from shadowspill.diagnostics.occupancy import attribute
from shadowspill.diagnostics.occupancy.page import page_data
from shadowspill.planner.serialization.simulation_result import simulation_result_value


def occupancy_data(
    plan: Plan,
    *,
    description: str = "",
    diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Attribute recorded schedules and physical leases; never simulate again."""
    communication = plan.communication_info
    rates = {}
    for direction in ("fetch", "evict"):
        solo = getattr(communication, f"{direction}_bytes_per_second")
        concurrent = getattr(communication, f"{direction}_concurrent_bytes_per_second")
        rates[f"{direction}_solo_bandwidth_bytes_per_second"] = solo
        rates[f"{direction}_concurrent_bandwidth_bytes_per_second"] = (
            solo if concurrent is None else concurrent
        )
        rates[f"{direction}_latency_ns"] = (
            getattr(communication, f"{direction}_latency_seconds") * 1e9
        )
    result = attribute(
        {
            "schedule": plan.schedule.to_dict(),
            "selections": [choice.to_dict() for choice in plan.selections],
            "simulation_result": simulation_result_value(plan.timeline),
            "simulation": {
                "spill_capacity_bytes": plan.memory_budgets.spill_bytes,
                "devices": [rates],
            },
            "admission_certificate": {
                "layout": None
                if plan.memory_layout is None
                else plan.memory_layout.to_dict()
            },
        },
        plan._costed_program.to_dict(),
        diagnostics=diagnostics,
    )
    data = page_data(result, plan=description or plan.program.name or "Plan")
    data["allocation"] = plan.summary
    data["simulation_only_reasons"] = plan.simulation_only_reasons
    return data
