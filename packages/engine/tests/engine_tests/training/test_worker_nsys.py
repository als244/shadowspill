"""The same one-shot Nsight window spans whole and split worker updates."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from engine_tests.training.test_worker_loop import SCOPE
from shadowspill.orchestration.capture import _nsys_runtime as module
from shadowspill.orchestration.capture._nsys_runtime import UpdateCapture
from shadowspill.orchestration.capture.nsys import NsysCapture
from shadowspill.orchestration.capture.progress import read_progress, write_progress
from shadowspill.training.worker.execution import WorkerExecution
from shadowspill.training.worker.protocol import WorkerOutputError
from shadowspill_client.buffers import BufferBudget


@pytest.fixture
def events(monkeypatch):
    result = []
    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(module.dist, "barrier", lambda: result.append("barrier"))
    monkeypatch.setattr(
        module.torch.cuda.profiler, "start", lambda: result.append("start")
    )
    monkeypatch.setattr(
        module.torch.cuda.profiler, "stop", lambda: result.append("stop")
    )

    @contextmanager
    def annotation(label):
        result.append(("enter", label))
        try:
            yield
        finally:
            result.append(("exit", label))

    monkeypatch.setattr(module.torch.cuda.nvtx, "range", annotation)
    return result


def worker(events, *, steps=2, fault=None, publish=None):
    capture = UpdateCapture(NsysCapture(start_step=2, steps=steps), publish=publish)

    def execute(**options):
        events.append(("execute", options["profiler_annotations"]))
        if fault == "execute":
            raise RuntimeError("device failed")
        return SimpleNamespace(result=lambda: None)

    def collect(*args):
        events.append("publish")
        if fault == "publish":
            raise OSError("output unavailable")

    executor = SimpleNamespace(
        run=execute,
        synchronize=lambda: events.append("sync"),
        close=lambda: events.append("close"),
    )
    execution = WorkerExecution(
        executor,
        SimpleNamespace(scope=SCOPE, primary="main"),
        BufferBudget(1),
        collect,
        capture=capture,
    )
    return execution, capture


def call(operation, updates, *, history="history", name=None):
    return SimpleNamespace(
        command=SimpleNamespace(
            scope=SCOPE,
            history_id=history,
            completed_updates=updates,
            accumulate_gradients=False,
            request=SimpleNamespace(
                operation=operation,
                runtime_trace=False,
                identity=SimpleNamespace(request_id=name or f"{operation}-{updates}"),
            ),
        ),
        plan="main",
        hyperparams={},
        input_loader=lambda _: None,
    )


def test_window_spans_every_backward_contribution_and_terminal_optimizer(events):
    execution, capture = worker(events)
    for operation, updates, name in (
        ("step", 0, "warmup"),
        ("forward", 1, "before"),
        ("forward_backward", 1, "part-one"),
        ("forward", 1, "evaluation"),
        ("forward_backward", 1, "part-two"),
        ("optim_step", 1, "optimizer"),
        ("step", 2, "whole"),
        ("step", 3, "after"),
        ("step", 1, "rewound"),
    ):
        bound = call(operation, updates, name=name)
        execution.prepare(bound)
        execution(bound)
    assert events.count("start") == events.count("stop") == 1
    assert events.count("barrier") == 4
    assert [
        item[1] for item in events if isinstance(item, tuple) and item[0] == "execute"
    ] == [False] * 2 + [True] * 5 + [False] * 2
    labels = [
        item[1] for item in events if isinstance(item, tuple) and item[0] == "enter"
    ]
    assert [label.split("/request=")[1].split("/")[0] for label in labels] == [
        "part-one",
        "evaluation",
        "part-two",
        "optimizer",
        "whole",
    ]
    assert all(label.endswith(f"/rank={SCOPE.rank}") for label in labels)
    stop = events.index("stop")
    assert events[stop - 2][0] == "exit" and events[stop - 3] == "sync"
    assert capture.state == "complete" and capture.completed_updates == 2
    assert capture.first_request == "part-one" and capture.last_request == "whole"
    assert capture.history_id == "history" and not capture.active


@pytest.mark.parametrize("fault", ["execute", "publish"])
def test_capture_keeps_confirmed_update_fact_separate_from_publication(events, fault):
    execution, capture = worker(events, steps=1, fault=fault)
    with pytest.raises(WorkerOutputError if fault == "publish" else RuntimeError):
        execution(call("step", 1))
    assert capture.state == ("complete" if fault == "publish" else "incomplete")
    assert capture.completed_updates == (1 if fault == "publish" else 0)
    assert events.count("stop") == (1 if fault == "publish" else 0)
    # An uncertain worker must not enter another distributed barrier on cleanup.
    capture.interrupt("worker closed")
    assert events.count("stop") == (1 if fault == "publish" else 0)


@pytest.mark.parametrize("change", ["history", "counter", "close", "gap"])
def test_interrupted_window_never_reports_complete_or_restarts(events, change):
    execution, capture = worker(events)
    execution(call("forward_backward", 1, name="first"))
    if change == "history":
        execution(call("forward", 1, history="new-history"))
    elif change == "counter":
        execution(call("forward", 0))
    elif change == "close":
        execution(call("close", 1))
    else:
        execution(call("step", 2))
    assert capture.state == "incomplete" and capture.error
    assert events.count("start") == events.count("stop") == 1
    execution(call("step", 1))
    assert events.count("start") == events.count("stop") == 1


def test_window_must_begin_at_first_selected_update_and_first_contribution(events):
    execution, capture = worker(events)
    execution(call("optim_step", 1))
    assert capture.state == "pending" and "start" not in events
    execution(call("step", 2))
    assert capture.state == "missing" and "start" not in events
    execution(call("step", 1))
    assert capture.state == "missing" and "start" not in events


def test_capture_requires_controller_history_before_worker_release(events):
    execution, _ = worker(events)
    with pytest.raises(ValueError, match="coordinator history"):
        execution.prepare(call("step", 1, history=None))
    assert not events


@pytest.mark.parametrize("fault", [False, True])
def test_bounded_progress_file_failures_do_not_change_completed_updates(
    tmp_path, events, fault
):
    path = tmp_path / "window.json"

    def publish(value):
        if fault and value.state == "complete":
            raise OSError("capture metadata disk full")
        write_progress(path, value)

    execution, capture = worker(events, steps=1, publish=publish)
    assert read_progress(path).state == "pending"
    execution(call("forward_backward", 1))
    assert read_progress(path).state == "running"
    execution(call("optim_step", 1))
    assert capture.state == "complete" and capture.completed_updates == 1
    assert events.count("start") == events.count("stop") == 1
    if fault:
        assert capture.publication_error == "capture metadata disk full"
        assert read_progress(path).state == "running"
    else:
        assert read_progress(path) == capture.progress
        assert capture.publication_error is None
    assert tuple(tmp_path.iterdir()) == (path,)
