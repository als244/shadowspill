"""Verified measurements in failures cannot imply an unconfirmed execution."""

from dataclasses import replace

import pytest

from shadowspill_client.protocol.codec import decode, encode
from shadowspill_client.protocol.results import Failure


def measured():
    return Failure(
        "publication_failed",
        "output unavailable",
        "completed",
        True,
        completed_updates=3,
        history_id="history",
        loss=1.125,
        metrics={"weight": 7},
    )


def test_failed_publication_measurements_roundtrip_and_own_their_values():
    values = {"weight": 7}
    failure = replace(measured(), metrics=values)
    values["weight"] = 0
    assert failure.metrics == {"weight": 7}
    assert decode(encode(failure)) == failure


@pytest.mark.parametrize(
    "changes",
    [
        {"execution_effect": "not_started"},
        {"execution_effect": "unknown", "state_valid": False},
        {"history_id": None},
        {"completed_updates": None},
        {"loss": float("nan")},
        {"metrics": {"bad": True}},
    ],
)
def test_invalid_failure_measurements_are_rejected(changes):
    with pytest.raises(ValueError):
        replace(measured(), **changes)
