from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import shadowspill.pytorch.callables as callable_module
from shadowspill.pytorch.callables import PlannedForward, PlannedTrainStep
from shadowspill.pytorch.runtime_adapter import RuntimeExecutionError


class _Signature:
    def validate(self, inputs: object) -> None:
        del inputs


class _FailingExecutor:
    def __init__(self, error: BaseException) -> None:
        self.error = error
        self.optimizer_restored = False

    def __call__(self, inputs: object) -> object:
        del inputs
        raise self.error

    def validate_invocation(self) -> None:
        pass

    def restore_optimizer_cpu(self) -> None:
        self.optimizer_restored = True


class _State:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.restored = False

    def restore_cpu_and_unregister(self) -> None:
        self.restored = True
        if self.fail:
            raise RuntimeError("state cleanup failed")


class _Runtime:
    def __init__(self, *, fail_release: bool = False) -> None:
        self.fail_release = fail_release
        self.adopted = False
        self.prepared_error: BaseException | None = None
        self.released = False

    def _adopt_plan(self, plan_handle: int) -> None:
        assert plan_handle == 77
        self.adopted = True

    def _prepare_failure_cleanup(
        self, error: BaseException, **kwargs: object
    ) -> None:
        del kwargs
        self.prepared_error = error

    def _release_plan(self, plan_handle: int) -> None:
        assert plan_handle == 77
        self.released = True
        if self.fail_release:
            raise RuntimeError("plan cleanup failed")


def test_forward_failure_attempts_every_cleanup_without_masking_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = RuntimeError("compiled task failed")
    executor = _FailingExecutor(original)
    state = _State(fail=True)
    runtime = _Runtime(fail_release=True)
    persistent_restored: list[object] = []
    monkeypatch.setattr(
        callable_module,
        "restore_persistent_object_ids",
        persistent_restored.append,
    )
    planned = PlannedForward(
        nn.Linear(1, 1),
        _Signature(),  # type: ignore[arg-type]
        executor,  # type: ignore[arg-type]
        state,  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        runtime,  # type: ignore[arg-type]
        77,
    )

    with pytest.raises(RuntimeError) as caught:
        planned([torch.ones(1)])

    assert caught.value is original
    assert planned._closed
    assert state.restored
    assert runtime.released
    assert runtime.prepared_error is original
    assert persistent_restored == [runtime]
    notes = getattr(original, "__notes__", ())
    assert any("restore model state" in note for note in notes)
    assert any("release runtime plan" in note for note in notes)
    planned.close()


def test_training_failure_restores_optimizer_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        callable_module,
        "validate_training_inputs",
        lambda inputs, signatures: None,
    )
    monkeypatch.setattr(
        callable_module,
        "restore_persistent_object_ids",
        lambda runtime: None,
    )
    original = RuntimeError("optimizer kernel failed")
    executor = _FailingExecutor(original)
    state = _State()
    runtime = _Runtime()
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    planned = PlannedTrainStep(
        model,
        (),
        executor,  # type: ignore[arg-type]
        state,  # type: ignore[arg-type]
        optimizer,
        object(),  # type: ignore[arg-type]
        runtime,  # type: ignore[arg-type]
        77,
    )

    with pytest.raises(RuntimeError) as caught:
        planned([])

    assert caught.value is original
    assert planned._closed
    assert executor.optimizer_restored
    assert state.restored
    assert runtime.released
    assert runtime.prepared_error is original


def test_cleanup_operations_raise_only_after_attempting_every_operation() -> None:
    completed: list[str] = []

    def fail() -> None:
        completed.append("fail")
        raise RuntimeError("first cleanup failure")

    def finish() -> None:
        completed.append("finish")

    with pytest.raises(RuntimeError, match="first cleanup failure"):
        callable_module._run_cleanup_operations(
            (("fail", fail), ("finish", finish)), primary_error=None
        )

    assert completed == ["fail", "finish"]


def test_runtime_failure_cleanup_is_claimed_once_across_nested_boundaries() -> None:
    error = RuntimeExecutionError("no-progress")

    assert error._begin_cleanup()
    assert not error._begin_cleanup()
