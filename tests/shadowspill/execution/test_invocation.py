"""The asynchronous result handle synchronizes once and retains failures."""

import pytest

from shadowspill.pytorch.invocation import InvocationResult


def test_invocation_result_synchronizes_once() -> None:
    synchronizations: list[str] = []
    resolved: list[InvocationResult[int]] = []
    failures: list[BaseException] = []
    invocation = InvocationResult(
        17,
        lambda: synchronizations.append("done"),
        on_resolved=resolved.append,
        on_failure=failures.append,
    )

    assert not invocation.resolved
    assert invocation.result() == 17
    assert invocation.wait() == 17
    assert invocation.resolved
    assert synchronizations == ["done"]
    assert resolved == [invocation]
    assert failures == []


def test_invocation_result_preserves_synchronization_failure() -> None:
    error = RuntimeError("completion failed")
    synchronizations = 0
    resolved: list[InvocationResult[int]] = []
    failures: list[BaseException] = []

    def synchronize() -> None:
        nonlocal synchronizations
        synchronizations += 1
        raise error

    invocation = InvocationResult(
        17,
        synchronize,
        on_resolved=resolved.append,
        on_failure=failures.append,
    )

    with pytest.raises(RuntimeError, match="completion failed") as first:
        invocation.result()
    with pytest.raises(RuntimeError, match="completion failed") as second:
        invocation.result()

    assert first.value is error
    assert second.value is error
    assert synchronizations == 1
    assert resolved == [invocation]
    assert failures == [error]
