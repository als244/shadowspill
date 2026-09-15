from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import shadowspill.pytorch.callables as callable_module
from shadowspill.pytorch.callables import PlannedForward, PlannedTrainStep
from shadowspill.pytorch.runtime_adapter import RuntimeExecutionError

from ._lifecycle import FakeRuntime, fake_plan_lifecycle


class _Signature:
    def validate(self, inputs: object) -> None:
        del inputs


class _Executor:
    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error
        self.optimizer_released = False

    # A training callable passes its step number; a forward callable has none.
    def __call__(self, inputs: object, step_number: int | None = None) -> object:
        del inputs, step_number
        if self.error is None:
            raise AssertionError("this executor was given no failure to raise")
        raise self.error

    def validate_invocation(self) -> None:
        pass

    def prepare_invocation(self, inputs: object) -> object:
        return inputs

    @property
    def optimizer_state(self) -> object:
        return SimpleNamespace(release=self._release_optimizer_state)

    def _release_optimizer_state(self) -> None:
        self.optimizer_released = True


class _State:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.restored = False

    def restore_cpu_and_unregister(self) -> None:
        self.restored = True
        if self.fail:
            raise RuntimeError("state cleanup failed")


def _planned_training(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[PlannedTrainStep, _Executor]:
    """Return a training callable whose every teardown operation succeeds."""

    fake_plan_lifecycle(monkeypatch, plan_handle=77)
    monkeypatch.setattr(
        callable_module,
        "restore_persistent_object_ids",
        lambda runtime: None,
    )
    executor = _Executor()
    model = nn.Linear(1, 1)
    planned = PlannedTrainStep(
        model,
        (),
        executor,  # type: ignore[arg-type]
        _State(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        FakeRuntime(),  # type: ignore[arg-type]
        77,
    )
    return planned, executor


def test_forward_failure_attempts_every_cleanup_without_masking_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = RuntimeError("indexed task failed")
    executor = _Executor(original)
    state = _State(fail=True)
    runtime = FakeRuntime(fail_release=True)
    fake_plan_lifecycle(monkeypatch, plan_handle=77)
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


def test_training_failure_releases_optimizer_state_and_closes(
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
    executor = _Executor(original)
    state = _State()
    runtime = FakeRuntime()
    fake_plan_lifecycle(monkeypatch, plan_handle=77)
    model = nn.Linear(1, 1)
    planned = PlannedTrainStep(
        model,
        (),
        executor,  # type: ignore[arg-type]
        state,  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        runtime,  # type: ignore[arg-type]
        77,
    )

    with pytest.raises(RuntimeError) as caught:
        planned([])

    assert caught.value is original
    assert planned._closed
    # A failed step publishes no optimizer update, and cleanup releases the
    # state with the plan either way.
    assert executor.optimizer_released
    assert state.restored
    assert runtime.released
    assert runtime.prepared_error is original
    with pytest.raises(RuntimeError, match="take the checkpoint before close"):
        planned.state_dict()


def test_close_releases_optimizer_state_and_refuses_a_later_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planned, executor = _planned_training(monkeypatch)

    planned.close()

    assert executor.optimizer_released
    with pytest.raises(RuntimeError, match="take the checkpoint before close"):
        planned.state_dict()
    with pytest.raises(RuntimeError, match="take the checkpoint before close"):
        planned.load_state_dict({})


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
