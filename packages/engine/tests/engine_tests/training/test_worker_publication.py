"""Only confirmed compute and released output views permit publication failure."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from engine_tests.training.test_worker_loop import SCOPE
from shadowspill.training.worker.execution import WorkerExecution
from shadowspill.training.worker.protocol import WorkerOutputError
from shadowspill_client.buffers import BufferBudget


@pytest.mark.parametrize("failure", [None, "disk", "copy", "before", "after"])
def test_output_failure_classification_requires_both_terminal_boundaries(failure):
    events = []

    class Output:
        def __del__(self):
            events.append("released")

    class Executor:
        synchronizations = 0

        def run(self, **kwargs):
            events.append("execute")
            return SimpleNamespace(result=Output)

        def synchronize(self):
            self.synchronizations += 1
            events.append("sync")
            if failure == ("before" if self.synchronizations == 1 else "after"):
                raise RuntimeError("device synchronization failed")

    def collect(call, value, budget):
        events.append("copy")
        if failure == "disk":
            raise OSError("output disk full")
        if failure == "copy":
            raise RuntimeError("device copy failed")
        return None

    call = SimpleNamespace(
        command=SimpleNamespace(
            scope=SCOPE,
            accumulate_gradients=False,
            request=SimpleNamespace(operation="step", runtime_trace=False),
        ),
        plan="main",
        hyperparams={},
        input_loader=lambda budget: None,
    )
    worker = WorkerExecution(
        Executor(),
        SimpleNamespace(scope=SCOPE, primary="main"),
        BufferBudget(1),
        collect,
    )
    if failure:
        with pytest.raises(RuntimeError) as error:
            worker(call)
        assert isinstance(error.value, WorkerOutputError) == (failure == "disk")
        if failure == "disk":
            assert events == ["execute", "sync", "copy", "released", "sync"]
    else:
        assert worker(call) is None
        assert events == ["execute", "sync", "copy", "released", "sync"]


def test_input_lease_is_released_before_the_same_budget_is_used_for_output():
    budget = BufferBudget(64, max_requests=1)
    events = []

    @contextmanager
    def loader(memory):
        with memory.reserve(64, timeout=0):
            events.append("input")
            yield object()
        events.append("input_closed")

    def execute(**kwargs):
        assert budget.used == (64, 1)
        return SimpleNamespace(result=lambda: object())

    def collect(call, value, memory):
        assert memory is budget and budget.used == (0, 0)
        with budget.reserve(64, timeout=0):
            events.append("output")

    call = SimpleNamespace(
        command=SimpleNamespace(
            scope=SCOPE,
            accumulate_gradients=False,
            request=SimpleNamespace(operation="step", runtime_trace=False),
        ),
        plan="main",
        hyperparams={},
        input_loader=loader,
    )
    worker = WorkerExecution(
        SimpleNamespace(run=execute, synchronize=lambda: None),
        SimpleNamespace(scope=SCOPE, primary="main"),
        budget,
        collect,
    )
    assert worker(call) is None
    assert events == ["input", "input_closed", "output"] and budget.used == (0, 0)
