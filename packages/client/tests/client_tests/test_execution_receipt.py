"""A portable completion fact never promises successful output publication."""

from dataclasses import replace

import pytest

from shadowspill_client.protocol.codec import decode, encode
from shadowspill_client.protocol.requests import RequestIdentity
from shadowspill_client.protocol.results import (
    Accumulation,
    ExecutionReceipt,
    OperationResult,
)


def receipt():
    return ExecutionReceipt(
        RequestIdentity("a", "run", 1, "writer", 1, "server"),
        "forward_backward",
        "completed",
        2,
        Accumulation(3, True, ("a",), 1),
    )


def test_completion_fact_roundtrips_independently_of_published_result():
    value = receipt()
    assert decode(encode(value)) == value
    result = OperationResult(
        value.identity,
        value.operation,
        "history",
        value.outcome,
        value.completed_updates,
        value.accumulation,
    )
    assert ExecutionReceipt.from_result(result) == value


@pytest.mark.parametrize(
    "change",
    [
        {"operation": "delete"},
        {"outcome": "running"},
        {"completed_updates": True},
        {"completed_updates": -1},
    ],
)
def test_invalid_completion_fact_is_rejected(change):
    with pytest.raises(ValueError):
        replace(receipt(), **change)
