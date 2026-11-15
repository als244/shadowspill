"""Portable immutable Plan metadata; no engine dependency."""

from dataclasses import replace

import pytest

from shadowspill_client.protocol.codec import decode, encode
from shadowspill_client.protocol.plans import PlanSummaryPage, PlanSummaryRow
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
