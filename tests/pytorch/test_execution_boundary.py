from __future__ import annotations

import weakref
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

from shadowspill.pytorch.execution.training import TrainingExecutor


class _RawOutputs:
    pass


class _AfterTaskHarness(TrainingExecutor):
    def __init__(self) -> None:
        self.released: weakref.ReferenceType[_RawOutputs] | None = None
        self.publication_observed_release = False

    @staticmethod
    def _before_task(_run: object, record: object) -> object:
        return SimpleNamespace(
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
    def _publish_output_generations(
        _prepared: object, _processed: object, _generations: object
    ) -> None:
        return None

    @staticmethod
    def _finish_task_cleanup(_prepared: object) -> None:
        return None

    @staticmethod
    def _abort_task(_prepared: object, _error: BaseException) -> None:
        return None

    @staticmethod
    def _finish_task_timing(_timing: object) -> None:
        return None


def test_after_task_releases_unadopted_outputs_before_runtime_actions() -> None:
    harness = _AfterTaskHarness()
    record = SimpleNamespace(trace_label="test")
    # Exercise the complete orchestration path: a local in _execute_task would
    # keep the result alive even if _after_task dropped its own parameter.
    TrainingExecutor._execute_task(harness, object(), record)  # type: ignore[arg-type]
    assert harness.publication_observed_release


class _TraceBoundaryBridge:
    def __init__(self, calls: list[object]) -> None:
        self.calls = calls
        self.statistics_value = object()

    def wait_idle(self) -> None:
        self.calls.append("wait_idle")

    def statistics(self) -> object:
        self.calls.append("statistics")
        return self.statistics_value

    def begin_runtime_trace(self, *, step_id: int) -> None:
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
        host_call_started_ns=0,
        host_startup_wait_ns=0,
        origin_event=SimpleNamespace(record=lambda _stream: calls.append("origin")),
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
        "wait_idle",
        "statistics",
        ("begin_runtime_trace", 4),
        "refresh_inputs",
    ]
    assert timing.statistics_before is bridge.statistics_value
