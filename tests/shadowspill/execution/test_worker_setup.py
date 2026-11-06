"""A failed worker setup releases what it opened and keeps its own error.

The cleanup runs while the setup error is being handled, where ExitStack drops
that error from the chain of anything a cleanup step raises. Once the runtime
has closed, the setup error itself is raised; a cleanup that could not close
the runtime raises, naming the setup error as its cause.
"""

from types import SimpleNamespace

import pytest

from shadowspill.execution import worker
from shadowspill.runtime import RuntimeConfigurationError


class _Runtime:
    def __init__(self, *, refuse: bool) -> None:
        self.refuse = refuse
        self._closed = False

    def close(self) -> None:
        if self.refuse:
            raise RuntimeConfigurationError(
                "cannot close Runtime while an active plan or plan setup owns it"
            )
        self._closed = True


class _LatchedPath:
    """A bound path whose plan steps a runtime that latched a failure refuses."""

    def close(self, runtime: object) -> None:
        raise RuntimeConfigurationError("handle plan close failed with status 41")


def _failing_setup(monkeypatch, runtime, error):
    """Bind one path on `runtime`, then fail setup with `error`."""

    def seal(candidates):
        raise error

    plan = SimpleNamespace(
        program=SimpleNamespace(parallelism=None), _accumulation_plan=None
    )
    monkeypatch.setattr(worker, "plan_inventory", lambda plan, plans: {"": plan})
    monkeypatch.setattr(worker, "plan_numerical_contracts", lambda plan: ())
    monkeypatch.setattr(worker, "worker_headroom", lambda candidates, headroom: 0)
    monkeypatch.setattr(worker, "worker_external_baseline", lambda candidates: 0)
    monkeypatch.setattr(worker, "open_worker", lambda *arguments: runtime)
    monkeypatch.setattr(worker, "resource_inventory", lambda programs: ({}, {}))
    monkeypatch.setattr(worker, "worker_streams", lambda candidates, device: None)
    monkeypatch.setattr(
        worker,
        "open_path",
        lambda runtime, plan, device: (SimpleNamespace(plan_handle=1), None),
    )
    monkeypatch.setattr(
        worker, "bind_path", lambda *arguments, **options: _LatchedPath()
    )
    monkeypatch.setattr(worker, "seal_paths", seal)
    return plan


def test_a_setup_error_is_raised_once_its_cleanup_closed_the_runtime(monkeypatch):
    runtime = _Runtime(refuse=False)
    error = ValueError("the bound paths could not be sealed")
    plan = _failing_setup(monkeypatch, runtime, error)

    with pytest.raises(ValueError) as raised:
        worker.create_paths(plan, 0, None)

    assert raised.value is error
    assert runtime._closed


def test_a_cleanup_that_could_not_close_the_runtime_names_the_setup_error(
    monkeypatch,
):
    runtime = _Runtime(refuse=True)
    error = ValueError("the bound paths could not be sealed")
    plan = _failing_setup(monkeypatch, runtime, error)

    with pytest.raises(RuntimeConfigurationError, match="active plan") as raised:
        worker.create_paths(plan, 0, None)

    assert raised.value.__cause__ is error
    assert not runtime._closed


def test_numeric_mismatch_precedes_opening_runtime_and_spill_pool(monkeypatch):
    import torch

    from shadowspill.pytorch.numerics import capture_numerical_settings

    contract = capture_numerical_settings()
    matmul = torch.backends.cuda.matmul
    monkeypatch.setattr(
        matmul,
        "allow_bf16_reduced_precision_reduction",
        not matmul.allow_bf16_reduced_precision_reduction,
    )
    runtime = _Runtime(refuse=False)
    plan = _failing_setup(monkeypatch, runtime, AssertionError("opened runtime"))
    monkeypatch.setattr(worker, "plan_numerical_contracts", lambda plan: (contract,))
    opened = []
    monkeypatch.setattr(worker, "open_worker", lambda *args: opened.append(args))
    with pytest.raises(RuntimeError, match="allow_bf16_reduced_precision_reduction"):
        worker.create_paths(plan, 0, None)
    assert not opened and not runtime._closed
