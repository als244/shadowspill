"""The task boundary's two halves and the invocation's opening, on fakes."""

from __future__ import annotations

import weakref
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest

import shadowspill.pytorch.execution.annotations as annotations_module
import shadowspill.pytorch.execution.timing as timing_module
import shadowspill.pytorch.execution.training.boundary as boundary_module
import shadowspill.pytorch.execution.training.publication as publication_module
from shadowspill.diagnostics.timing import InvocationTimelines
from shadowspill.pytorch.execution.annotations import TaskBoundaryAnnotations
from shadowspill.pytorch.execution.timing import ExecutionTiming
from shadowspill.pytorch.execution.training import TrainingExecutor
from shadowspill.pytorch.execution.training.boundary import execute_task
from shadowspill.pytorch.execution.training.publication import after_task
from tests.shadowspill.runtime._timing import TimingLibrary, install


class _CallLog:
    """What a fake bridge records; the bridge functions are patched to write here."""

    def __init__(self) -> None:
        self.calls: list[object] = []
        self.statistics_value = object()
        # Timing markers come from the runtime, so a bridge names one.
        self.runtime = SimpleNamespace(_runtime_handle=0)

    def wait_until_idle(self) -> None:
        self.calls.append("wait_plan_idle")

    def require_empty_layout(self) -> None:
        self.calls.append("require_empty_layout")


class _RawOutputs:
    pass


def _executor(calls: list[object] | None = None) -> TrainingExecutor:
    """A bare executor with the attributes the boundary functions read."""

    executor = object.__new__(TrainingExecutor)
    # Annotations stay off unless a test turns them on, so the bridge is never
    # asked anything; timing finishes tasks into the call log.
    executor._task_annotations = TaskBoundaryAnnotations(cast(Any, _CallLog()))
    executor.timing = cast(
        Any,
        SimpleNamespace(
            finish_task=lambda _timing: (
                calls.append("finish_timing") if calls is not None else None
            )
        ),
    )
    return executor


def test_after_task_releases_unadopted_outputs_before_runtime_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = _executor()
    released: list[weakref.ReferenceType[_RawOutputs]] = []
    observed: list[bool] = []

    def run_compiled_task(_executor: object, _prepared: object) -> _RawOutputs:
        value = _RawOutputs()
        released.append(weakref.ref(value))
        return value

    def publish_task_to_runtime(
        _executor: object,
        _prepared: object,
        _processed: object,
        _dematerialized: object,
    ) -> None:
        observed.append(released[0]() is None)

    record = SimpleNamespace(
        trace_label="test",
        released_ephemeral=(),
        task=SimpleNamespace(task_id="task"),
    )
    run = SimpleNamespace(lowered=SimpleNamespace(optimizer_task_id="optimizer"))
    monkeypatch.setattr(
        boundary_module,
        "before_task",
        lambda _executor, run, record: SimpleNamespace(
            run=run, record=record, timing=None
        ),
    )
    monkeypatch.setattr(boundary_module, "run_compiled_task", run_compiled_task)
    monkeypatch.setattr(
        publication_module,
        "prepare_task_publication",
        lambda _executor, _prepared, _raw: (SimpleNamespace(outputs=()), ()),
    )
    monkeypatch.setattr(
        publication_module, "publish_task_to_runtime", publish_task_to_runtime
    )
    monkeypatch.setattr(
        publication_module,
        "publish_frontend_bindings",
        lambda _executor, _prepared, _processed: None,
    )
    monkeypatch.setattr(
        publication_module, "finish_task_cleanup", lambda _executor, _prepared: None
    )
    # Exercise the complete orchestration path: a local in execute_task would
    # keep the result alive even if after_task dropped its own parameter.
    execute_task(executor, cast(Any, run), cast(Any, record))
    assert observed == [True]


def test_after_task_annotation_and_timing_cover_the_complete_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    executor = _executor(calls)
    executor._task_annotations = cast(
        Any,
        SimpleNamespace(
            enabled=True,
            begin=lambda _label: (calls.append("range_enter"), 1)[1],
            end=lambda _range_id: calls.append("range_exit"),
        ),
    )
    monkeypatch.setattr(
        publication_module,
        "prepare_task_publication",
        lambda _executor, _prepared, _raw: (
            calls.append("process_outputs"),
            (SimpleNamespace(outputs=()), ()),
        )[1],
    )
    monkeypatch.setattr(
        publication_module,
        "publish_task_to_runtime",
        lambda _executor, _prepared, _processed, _dematerialized: calls.append(
            "runtime_after_task"
        ),
    )
    monkeypatch.setattr(
        publication_module,
        "publish_frontend_bindings",
        lambda _executor, _prepared, _processed: calls.append("publish_frontend"),
    )
    monkeypatch.setattr(
        publication_module,
        "finish_task_cleanup",
        lambda _executor, _prepared: calls.append("cleanup"),
    )
    prepared = SimpleNamespace(
        run=SimpleNamespace(lowered=SimpleNamespace(optimizer_task_id="optimizer")),
        record=SimpleNamespace(
            trace_label="test",
            released_ephemeral=("temporary",),
            task=SimpleNamespace(task_id="task"),
        ),
        timing=object(),
    )

    after_task(executor, cast(Any, prepared), object())

    assert calls == [
        "range_enter",
        "process_outputs",
        "runtime_after_task",
        "publish_frontend",
        "cleanup",
        "range_exit",
        "finish_timing",
    ]


def test_runtime_trace_begins_after_prior_invocation_is_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = _CallLog()
    calls = bridge.calls

    def statistics(bridge: _CallLog) -> object:
        calls.append("statistics")
        return bridge.statistics_value

    monkeypatch.setattr(timing_module, "statistics", statistics)
    monkeypatch.setattr(
        timing_module,
        "begin_runtime_trace",
        lambda bridge, *, step_id, origin=None: calls.append(
            ("begin_runtime_trace", step_id)
        ),
    )
    run = object()
    harness = object.__new__(TrainingExecutor)
    harness._bridge = bridge  # type: ignore[assignment]
    harness._invocations = 3
    install(monkeypatch, TimingLibrary())
    harness.timing = ExecutionTiming(cast(Any, bridge), ())
    # Every invocation records its timeline against the fake library, which
    # keeps the call log to the boundary's own steps.
    harness.timing._timelines = InvocationTimelines(0)
    harness._initial = None
    harness.optimizer_state = cast(Any, SimpleNamespace(initialized=True))
    harness._recurrent = run  # type: ignore[assignment]
    harness._active_run = run  # type: ignore[assignment]
    harness._state = SimpleNamespace(  # type: ignore[assignment]
        refresh_inputs=lambda _inputs: calls.append("refresh_inputs")
    )
    timing = SimpleNamespace(
        dispatch_call_started_ns=0,
        prior_invocation_drain_ns=0,
        origin_event=SimpleNamespace(
            record=lambda _stream: calls.append("origin"), cuda_event=0
        ),
        statistics_before=None,
    )
    # The executor records the origin of whatever is armed, which is the
    # record its caller passes down.
    harness.timing.armed = cast(Any, timing)

    with patch(
        "shadowspill.pytorch.execution.training.torch.cuda.current_stream",
        return_value=SimpleNamespace(cuda_stream=11),
    ):
        # The caller numbers the step; a restored checkpoint makes it differ
        # from this process's invocation count, and the trace follows the caller.
        selected = TrainingExecutor._begin_invocation(harness, (), cast(Any, timing), 7)

    assert selected is run
    assert calls == [
        "origin",
        "wait_plan_idle",
        "statistics",
        ("begin_runtime_trace", 7),
        "refresh_inputs",
        "require_empty_layout",
    ]
    assert timing.statistics_before is bridge.statistics_value


def test_task_boundary_annotations_are_shared_and_default_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A range costs nothing until something turns annotations on.

    The bridge is asked for a provider range only on the enabled path, so the
    disabled body runs without the bridge being touched at all.
    """

    calls: list[object] = []
    monkeypatch.setattr(
        annotations_module,
        "set_profiler_annotations",
        lambda _bridge, enabled: calls.append(("enabled", enabled)),
    )
    monkeypatch.setattr(
        annotations_module,
        "profile_range_begin",
        lambda _bridge, name: (calls.append(("begin", name)), 17)[1],
    )
    monkeypatch.setattr(
        annotations_module,
        "profile_range_end",
        lambda _bridge, range_id: calls.append(("end", range_id)),
    )
    annotations = TaskBoundaryAnnotations(cast(Any, _CallLog()))

    assert not annotations.enabled
    with annotations.range("ignored"):
        calls.append("disabled_body")
    annotations.set_enabled(True)
    with annotations.range("task"):
        calls.append("enabled_body")

    assert calls == [
        "disabled_body",
        ("enabled", True),
        ("begin", "task"),
        "enabled_body",
        ("end", 17),
    ]
