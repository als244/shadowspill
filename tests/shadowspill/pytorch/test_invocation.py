from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
import torch.nn as nn

import shadowspill.pytorch.callables as callable_module
from shadowspill.pytorch.callables import PlannedForward
from shadowspill.pytorch.invocation import InvocationResult


class _Signature:
    def validate(self, inputs: object) -> None:
        del inputs


class _Executor:
    def __init__(self) -> None:
        self.synchronizations = 0
        self.invocations = 0

    def prepare_invocation(self, inputs: object) -> object:
        return inputs

    def validate_invocation(self) -> None:
        pass

    def __call__(self, inputs: object) -> object:
        self.invocations += 1
        return inputs

    def record_invocation_completion(self) -> Callable[[], None]:
        def synchronize() -> None:
            self.synchronizations += 1

        return synchronize


class _State:
    def __init__(self) -> None:
        self.restored = False

    def restore_cpu_and_unregister(self) -> None:
        self.restored = True


class _Runtime:
    def __init__(self) -> None:
        self.released = False

    def _adopt_plan(self, plan_handle: int) -> None:
        assert plan_handle == 7

    def _release_plan(self, plan_handle: int) -> None:
        assert plan_handle == 7
        self.released = True

    def _prepare_failure_cleanup(
        self, error: BaseException, **kwargs: object
    ) -> None:
        del error, kwargs


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


def test_submitted_forward_requires_explicit_result_before_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = _Executor()
    state = _State()
    runtime = _Runtime()
    monkeypatch.setattr(
        callable_module,
        "restore_persistent_object_ids",
        lambda owner: None,
    )
    planned = PlannedForward(
        nn.Linear(1, 1),
        _Signature(),  # type: ignore[arg-type]
        executor,  # type: ignore[arg-type]
        state,  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        runtime,  # type: ignore[arg-type]
        7,
    )
    planned._release_executor = lambda: None  # type: ignore[method-assign]

    submitted = planned.submit([torch.ones(1)])
    assert executor.synchronizations == 0
    with pytest.raises(RuntimeError, match="resolve the preceding"):
        planned([torch.ones(1)])

    output = submitted.result()
    assert isinstance(output, list)
    assert executor.synchronizations == 1
    planned([torch.ones(1)])
    assert executor.invocations == 2

    planned.close()
    assert state.restored
    assert runtime.released


def test_close_resolves_a_pending_submission_without_recursive_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = _Executor()
    state = _State()
    runtime = _Runtime()
    monkeypatch.setattr(
        callable_module,
        "restore_persistent_object_ids",
        lambda owner: None,
    )
    planned = PlannedForward(
        nn.Linear(1, 1),
        _Signature(),  # type: ignore[arg-type]
        executor,  # type: ignore[arg-type]
        state,  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        runtime,  # type: ignore[arg-type]
        7,
    )
    planned._release_executor = lambda: None  # type: ignore[method-assign]

    submitted = planned.submit([])
    planned.close()

    assert submitted.resolved
    assert executor.synchronizations == 1
    assert state.restored
    assert runtime.released
