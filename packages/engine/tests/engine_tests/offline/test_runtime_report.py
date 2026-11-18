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
    # A trace recorded without its request's tokens states no token rate.
    assert measured["summary"]["tokens_per_second"] is None
    assert predicted["summary"]["tokens_per_second"] is None
    assert measured["allocation"] == predicted["allocation"] == plan.summary
    assert "</script><script>unsafe" not in html
    assert "Memory occupancy is reconstructed" in html
    with pytest.raises(ValueError, match="admitted report bytes"):
        runtime_html(plan, evidence(plan), invocation=invocation, max_bytes=100)


def test_each_view_states_its_own_tokens_over_its_own_time():
    plan = plan_program(
        profiled(),
        memory_budgets=MemoryBudgets(100, 64),
        communication_info=communication(),
        search_options=options(),
    )
    invocation = {
        "operation": "step",
        "rank": 0,
        "plan": "plan",
        "batch_tokens": 900,
        "scheduled_tokens": 1024,
    }
    html = runtime_html(
        plan, evidence(plan), invocation=invocation, max_bytes=1 << 20
    ).decode()
    payload = json.loads(
        html.split('type="application/json">', 1)[1].split("</script>", 1)[0]
    )
    measured, predicted = [view["data"]["summary"] for view in payload["views"]]
    # The trace is timed against what the request supplied; the saved Plan is
    # priced for what its candidate schedules, padding included.
    assert measured["tokens_per_second"] == pytest.approx(900 / measured["seconds"])
    assert predicted["tokens_per_second"] == pytest.approx(1024 / predicted["seconds"])


def test_every_view_carries_its_plans_alternatives_and_the_trace_its_times():
    plan = plan_program(
        profiled(),
        memory_budgets=MemoryBudgets(100, 64),
        communication_info=communication(),
        search_options=options(),
    )
    html = runtime_html(
        plan,
        evidence(plan),
        invocation={"operation": "step", "rank": 0, "plan": "plan"},
        max_bytes=1 << 20,
    ).decode()
    payload = json.loads(
        html.split('type="application/json">', 1)[1].split("</script>", 1)[0]
    )
    measured, predicted = [view["data"]["alternatives"] for view in payload["views"]]
    rows = [
        row
        for found in (measured, predicted)
        for row in (
            *found["single"],
            *(
                task
                for form in found["forms"]
                for option in form["options"]
                for task in option["tasks"]
            ),
        )
    ]
    assert rows and len(rows) % 2 == 0
    # The same tasks on both views; only the trace observed any time.
    half = len(rows) // 2
    assert [row.keys() for row in rows[:half]] == [row.keys() for row in rows[half:]]
    assert any(row["measured"] for row in rows[:half])
    assert not any(row["measured"] for row in rows[half:])
    assert measured["options"] == predicted["options"]
    chosen = {choice.group_id: choice.option_id for choice in plan.selections}
    assert sum(
        option["chosen"] for form in predicted["forms"] for option in form["options"]
    ) == len(chosen)
