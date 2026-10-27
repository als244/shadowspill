"""Recording is opt-in, includes terminal work, and recovers failed setup."""

from types import SimpleNamespace
from typing import Any, cast

import pytest

import shadowspill.execution.recording.timing as timing_module
import shadowspill.runtime.plan.annotations as annotations_module
from shadowspill.execution.recording.timing import ExecutionTiming
from shadowspill.runtime.plan.annotations import TaskBoundaryAnnotations
from shadowspill.runtime.plan.tracing import TraceCapacities
from tests.shadowspill.runtime._timing import TimingLibrary, install

_CAPACITIES = TraceCapacities(events=1, allocation_events=1)


def _bridge():
    return SimpleNamespace(runtime=SimpleNamespace(_runtime_handle=0))


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
    annotations = TaskBoundaryAnnotations(cast(Any, _bridge()))

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


def test_final_cycle_includes_required_terminal_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library = install(monkeypatch, TimingLibrary())
    bridge = SimpleNamespace(
        runtime=SimpleNamespace(_runtime_handle=0),
        wait_until_idle=lambda: library.clock.advance(4.0),
    )
    timing = ExecutionTiming(cast(Any, bridge), (), _CAPACITIES)
    stream = SimpleNamespace(cuda_stream=11)
    timing.begin_invocation(1, cast(Any, stream))
    timing.record_compute_start(cast(Any, stream))
    library.clock.advance(1.0)
    timing.record_compute_end(cast(Any, stream))
    monkeypatch.setattr(timing_module.torch.cuda, "current_stream", lambda: stream)
    timing.mark_cycle_end()
    (result,) = timing.invocation_timings()
    assert result.cycle_seconds == pytest.approx(0.005)
    assert result.selected_span_seconds == pytest.approx(0.001)
    assert result.exposed_tail_seconds == pytest.approx(0.004)
    timing.release()


def test_trace_preparation_is_reusable_and_recovers_from_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library = install(monkeypatch, TimingLibrary())
    timing = ExecutionTiming(cast(Any, _bridge()), ("task_000000",), _CAPACITIES)
    attempts = []

    def prepare(*args: object, **kwargs: object) -> None:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("trace allocation failed")

    monkeypatch.setattr(timing_module, "prepare_runtime_trace", prepare)
    with pytest.raises(RuntimeError, match="trace allocation failed"):
        timing.prepare()
    assert len(library.released) == 7  # three invocation and four task markers
    timing.prepare()
    timing.prepare()
    assert len(attempts) == 2
    timing.release()
    assert len(library.released) == 14
