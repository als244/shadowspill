from __future__ import annotations

import weakref
from contextlib import nullcontext
from types import SimpleNamespace

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
