"""Measured HTML and the matching simulation share the existing occupancy renderer."""

import json

import pytest

from engine_tests.offline._example import communication, options, profiled
from shadowspill import MemoryBudgets, plan_program
from shadowspill.reporting.runtime import runtime_html


def evidence(plan):
    return {
        "tasks": {
            row.task_id: {
                "task_id": row.task_id,
                "compute": {
                    "compute_started_at_seconds": row.start_ns * 3e-9,
                    "compute_finished_at_seconds": row.end_ns * 3e-9,
                },
            }
            for row in plan.timeline.task_intervals
        },
        "transfers": {
            direction: {
                str(row.sequence): {
                    "sequence": row.sequence,
                    "alias_group_id": row.alias_group_id,
                    "transfer_id": f"{direction}_{row.sequence}",
                    "bytes": row.bytes,
                    "lane": {
                        "lane_started_at_seconds": row.start_ns * 3e-9,
                        "lane_finished_at_seconds": row.end_ns * 3e-9,
                    },
                }
                for row in plan.timeline.transfer_intervals
                if row.direction == direction
            }
            for direction in ("fetch", "evict")
        },
    }


def test_runtime_html_uses_measured_clocks_and_preserves_saved_plan_identity():
    plan = plan_program(
        profiled(),
        memory_budgets=MemoryBudgets(100, 64),
        communication_info=communication(),
        search_options=options(),
    )
    invocation = {
        "operation": "step",
        "rank": 0,
        "plan": "</script><script>unsafe</script>",
    }
    html = runtime_html(
        plan, evidence(plan), invocation=invocation, max_bytes=1 << 20
    ).decode()
    payload = json.loads(
        html.split('type="application/json">', 1)[1].split("</script>", 1)[0]
    )
    measured, predicted = [view["data"] for view in payload["views"]]
    assert payload["rows"][0]["values"] == invocation
    assert measured["view"] == "traced" and predicted["view"] == "simulated"
    assert measured["summary"]["seconds"] == pytest.approx(
        3 * predicted["summary"]["seconds"]
    )
    assert measured["compute"] != predicted["compute"]
    assert measured["allocation"] == predicted["allocation"] == plan.summary
    assert "</script><script>unsafe" not in html
    assert "Memory occupancy is reconstructed" in html
    with pytest.raises(ValueError, match="admitted report bytes"):
        runtime_html(plan, evidence(plan), invocation=invocation, max_bytes=100)
