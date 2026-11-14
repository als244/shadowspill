"""Only confirmed compute and released output views permit publication failure."""

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

    def collect(value):
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
            request=SimpleNamespace(runtime_trace=False),
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
