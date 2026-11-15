"""Metric pages preserve scalar values, request identity and journal cursors."""

from dataclasses import replace

import pytest

from shadowspill_client.protocol.codec import decode, encode
from shadowspill_client.protocol.metrics import MetricPage, MetricRecord
from shadowspill_client.protocol.requests import RequestIdentity


def metric():
    return MetricRecord(
        9,
        RequestIdentity("step", "run", 1, "writer", 1, "server"),
        "step",
        "completed",
        "history",
        1,
        True,
        "completed",
        loss=7.25,
        metrics={"count": 3},
    )


def test_metric_roundtrip_freezes_values_and_keeps_loss_separate():
    values = {"count": 3, "loss": 0.125}
    row = replace(metric(), metrics=values)
    values["count"] = 9
    page = MetricPage("session", "run", 0, 10, [row], next_cursor=9)
    assert decode(encode(page)) == page
    assert row.metrics == {"count": 3, "loss": 0.125} and row.loss == 7.25
    assert decode(encode(row)) == row


@pytest.mark.parametrize(
    "changes",
    [
        {"position": 0},
        {"position": True},
        {"position": 1 << 63},
        {"loss": float("nan")},
        {"metrics": {"count": True}},
        {"metrics": {"x": []}},
        {"operation": "unknown"},
        {"outcome": "accepted"},
        {"history_id": None},
        {"completed_updates": None},
        {"execution_effect": "unknown"},
        {"failure_code": "oops"},
    ],
)
def test_invalid_scalar_or_execution_facts_are_rejected(changes):
    with pytest.raises((TypeError, ValueError)):
        replace(metric(), **changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"after": 10},
        {"through": 8},
        {"next_cursor": 8},
        {"run_id": "other"},
        {"records": [metric(), metric()]},
        {"records": [], "next_cursor": 9},
    ],
)
def test_invalid_page_scope_order_and_cursor_are_rejected(changes):
    with pytest.raises(ValueError):
        replace(MetricPage("session", "run", 0, 10, [metric()]), **changes)
