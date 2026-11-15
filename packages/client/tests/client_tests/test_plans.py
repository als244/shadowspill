"""Portable immutable Plan metadata; no engine dependency."""

from dataclasses import replace

import pytest

from shadowspill_client.protocol.codec import decode, encode
from shadowspill_client.protocol.plans import (
    PlanSummaryPage,
    PlanSummaryRow,
    SavedPlans,
)
from shadowspill_client.protocol.requests import ArtifactRef


def test_summary_wire_roundtrip_is_bounded_and_owned():
    values = {"seconds": 0.5, "choices": [1, 2]}
    row = PlanSummaryRow("worker_plan", values)
    page = PlanSummaryPage(
        ArtifactRef("plans", "a" * 64, 123), "b" * 64, 0, 1, 1, 1, [row]
    )
    values["choices"].append(3)
    assert page.rows[0].values["choices"] == (1, 2)
    assert decode(encode(page)) == page
    for options in (
        {"rows": []},
        {"rows": [row] * 33},
        {"next_page": 3},
        {"configuration_sha256": "invalid"},
    ):
        with pytest.raises(ValueError):
            replace(page, **options)


def test_named_plans_preserve_unicode_names_and_bound_context():
    saved = SavedPlans(
        "session", "run", "模型 / best + 1", ArtifactRef("plans", "a" * 64, 3)
    )
    assert decode(encode(saved)) == saved
    for options in (
        {"session_id": "../other"},
        {"run_id": "a/b"},
        {"name": ""},
        {"name": " leading"},
        {"name": "x" * 257},
        {"name": "\ud800"},
        {"plans": None},
    ):
        with pytest.raises((TypeError, ValueError)):
            replace(saved, **options)
