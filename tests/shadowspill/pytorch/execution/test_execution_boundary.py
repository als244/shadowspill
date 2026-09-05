from __future__ import annotations

import weakref
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

from shadowspill.pytorch.execution.annotations import TaskBoundaryAnnotations
from shadowspill.pytorch.execution.training import TrainingExecutor


class _RawOutputs:
    pass


class _AfterTaskHarness(TrainingExecutor):
    def __init__(self) -> None:
        self.released: weakref.ReferenceType[_RawOutputs] | None = None
        self.publication_observed_release = False
        self._task_annotations = TaskBoundaryAnnotations(cast(Any, _AnnotationBridge()))

    @staticmethod
    def _before_task(run: object, record: object) -> object:
        return SimpleNamespace(
            run=run,
            record=record,
            timing=None,
        )

    def _run_compiled_task(self, _prepared: object) -> _RawOutputs:
        value = _RawOutputs()
        self.released = weakref.ref(value)
        return value

    @staticmethod
    def _profile_range(_label: str) -> object:
        return nullcontext()

    @staticmethod
    def _prepare_task_publication(
        _prepared: object, _raw_outputs: object
    ) -> tuple[object, tuple[object, ...]]:
        return SimpleNamespace(outputs=()), ()

    def _publish_task_to_runtime(
        self, _prepared: object, _processed: object, _dematerialized: object
    ) -> tuple[int, ...]:
        self.publication_observed_release = (
            self.released is not None and self.released() is None
        )
        return ()

    @staticmethod
    def _publish_frontend_bindings(_prepared: object, _processed: object) -> None:
        return None

    @staticmethod
    def _finish_task_cleanup(_prepared: object) -> None:
        return None

    @staticmethod
    def _abort_task(_prepared: object) -> None:
        return None

    @staticmethod
    def _finish_task_timing(_timing: object) -> None:
        return None


def test_after_task_releases_unadopted_outputs_before_runtime_actions() -> None:
    harness = _AfterTaskHarness()
    record = SimpleNamespace(
        trace_label="test",
        released_ephemeral=(),
        task=SimpleNamespace(task_id="task"),
    )
    run = SimpleNamespace(lowered=SimpleNamespace(optimizer_task_id="optimizer"))
    # Exercise the complete orchestration path: a local in _execute_task would
    # keep the result alive even if _after_task dropped its own parameter.
    TrainingExecutor._execute_task(harness, run, record)  # type: ignore[arg-type]
    assert harness.publication_observed_release


class _TimedAfterTaskHarness(_AfterTaskHarness):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []
        self._task_annotations = cast(
            Any,
            SimpleNamespace(
                enabled=True,
                begin=self._begin_annotation,
                end=self._end_annotation,
            ),
        )

    def _begin_annotation(self, _label: str) -> int:
        self.calls.append("range_enter")
        return 1

    def _end_annotation(self, _range_id: int) -> None:
        self.calls.append("range_exit")

    def _prepare_task_publication(
        self, _prepared: object, _raw_outputs: object
    ) -> tuple[object, tuple[object, ...]]:
        self.calls.append("process_outputs")
        return SimpleNamespace(outputs=()), ()

    def _publish_task_to_runtime(
        self, _prepared: object, _processed: object, _dematerialized: object
    ) -> tuple[int, ...]:
        self.calls.append("runtime_after_task")
        return ()

    def _publish_frontend_bindings(self, _prepared: object, _processed: object) -> None:
        self.calls.append("publish_frontend")

    def _finish_task_cleanup(self, _prepared: object) -> None:
        self.calls.append("cleanup")

    def _finish_task_timing(self, _timing: object) -> None:
        self.calls.append("finish_timing")


def test_after_task_annotation_and_timing_cover_the_complete_boundary() -> None:
    harness = _TimedAfterTaskHarness()
    prepared = SimpleNamespace(
        run=SimpleNamespace(lowered=SimpleNamespace(optimizer_task_id="optimizer")),
        record=SimpleNamespace(
            trace_label="test",
            released_ephemeral=("temporary",),
            task=SimpleNamespace(task_id="task"),
        ),
        timing=object(),
    )

    TrainingExecutor._after_task(harness, prepared, object())  # type: ignore[arg-type]

    assert harness.calls == [
        "range_enter",
        "process_outputs",
        "runtime_after_task",
        "publish_frontend",
        "cleanup",
        "range_exit",
        "finish_timing",
    ]


class _AnnotationBridge:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def set_profiler_annotations(self, enabled: bool) -> None:
        self.calls.append(("enabled", enabled))

    def profile_range_begin(self, name: str) -> int:
        self.calls.append(("begin", name))
        return 17

    def profile_range_end(self, range_id: int) -> None:
        self.calls.append(("end", range_id))


def test_task_boundary_annotations_are_shared_and_default_off() -> None:
    bridge = _AnnotationBridge()
    annotations = TaskBoundaryAnnotations(cast(Any, bridge))

    with annotations.range("ignored"):
        bridge.calls.append("disabled_body")
    annotations.set_enabled(True)
    with annotations.range("task"):
        bridge.calls.append("enabled_body")

    assert bridge.calls == [
        "disabled_body",
        ("enabled", True),
        ("begin", "task"),
        "enabled_body",
        ("end", 17),
    ]


class _TraceBoundaryBridge:
    def __init__(self, calls: list[object]) -> None:
        self.calls = calls
        self.statistics_value = object()

    def wait_plan_idle(self) -> None:
        self.calls.append("wait_plan_idle")

    def statistics(self) -> object:
        self.calls.append("statistics")
        return self.statistics_value

    def begin_runtime_trace(
        self, *, step_id: int, origin_event_handle: int | None = None
    ) -> None:
        self.calls.append(("begin_runtime_trace", step_id))


def test_runtime_trace_begins_after_prior_invocation_is_idle() -> None:
    calls: list[object] = []
    bridge = _TraceBoundaryBridge(calls)
    run = object()
    harness = object.__new__(TrainingExecutor)
    harness._bridge = bridge  # type: ignore[assignment]
    harness._invocations = 3
    harness._initial = None
    harness._optimizer_state_initialized = True
    harness._recurrent = run  # type: ignore[assignment]
    harness._active_run = run  # type: ignore[assignment]
    harness._trace_label_run = run  # type: ignore[assignment]
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

    with patch(
        "shadowspill.pytorch.execution.training.torch.cuda.current_stream",
        return_value=object(),
    ):
        selected = TrainingExecutor._begin_invocation(harness, (), cast(Any, timing))

    assert selected is run
    assert calls == [
        "origin",
        "wait_plan_idle",
        "statistics",
        ("begin_runtime_trace", 4),
        "refresh_inputs",
    ]
    assert timing.statistics_before is bridge.statistics_value
