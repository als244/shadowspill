"""A failed run is terminal: one close releases the plan and the runtime.

The CPU cases drive the real close path, Executor through ExecutionPath and
release_plan to Runtime.close, with only the C entry points replaced. Once the
runtime has closed, close returns, so a caller that closes in a `finally` sees
the error that stopped the run. Only a close the runtime refuses raises, and
what it raises names the failure.
"""

import os
import subprocess
import sys
import threading
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from shadowspill.build import build_forward
from shadowspill.errors import AdmissionError
from shadowspill.execution import Executor
from shadowspill.execution.paths import ExecutionPath
from shadowspill.pytorch.invocation import InvocationResult
from shadowspill.runtime import Runtime, RuntimeConfigurationError
from shadowspill.runtime.plan import lifecycle
from shadowspill.runtime.plan.bridge import RuntimeBridge
from shadowspill.runtime.plan.lifecycle import PlanMemory
from shadowspill.runtime.plan.objects import PlanObjects
from shadowspill.status import Status
from tests.shadowspill.execution.helpers import save_plan

PLAN = 0xBEEF
ALIASES = {"alias_000001": 101, "alias_000002": 102}


class _Libraries:
    """The neutral and adapter entry points a close reaches.

    A runtime that latched a failure refuses the task, object and plan steps
    with that status. Closing the runtime still releases everything and
    returns the status it observed, as the C close does when latched.
    """

    def __init__(self, latched: int = 0) -> None:
        self.latched = latched
        self.calls: list[str] = []
        self.published = True
        self.caller_outputs = lambda: False

    def shadowspill_plan_clear_tasks(self, handle):
        self.calls.append("plan_clear_tasks")
        return self.latched

    def shadowspill_unregister_object(self, runtime, object_id):
        self.calls.append(f"unregister_object({object_id})")
        return self.latched

    def shadowspill_plan_close(self, handle):
        self.calls.append("plan_close")
        return self.latched

    def shadowspill_plan_destroy(self, handle):
        self.calls.append("plan_destroy")

    def shadowspill_pytorch_allocator_failure(self, pointer):
        failure = pointer._obj
        failure.status = failure.runtime.status = self.latched
        return self.latched

    def shadowspill_pytorch_allocator_close(self):
        self.calls.append("allocator_close")
        if self.caller_outputs():
            return Status.INVALID_STATE
        self.published = False
        return self.latched

    def shadowspill_pytorch_runtime_handle(self, pointer):
        return Status.OK if self.published else Status.CLOSED


@pytest.fixture
def synchronized(monkeypatch):
    devices = []
    monkeypatch.setattr(torch.cuda, "synchronize", devices.append)
    return devices


def _executor(libraries, monkeypatch, failure=None):
    monkeypatch.setattr(lifecycle, "runtime_library", lambda: libraries)
    runtime = Runtime.__new__(Runtime)
    runtime._lock = threading.RLock()
    runtime._closed = False
    runtime._unusable_reason = None
    runtime._active_plan_handles = {PLAN}
    runtime._pending_plan_handle = None
    runtime._active_object_references = 0
    runtime._installed = SimpleNamespace(library=libraries)
    runtime._runtime_handle = 1

    objects = PlanObjects.__new__(PlanObjects)
    objects.runtime = runtime
    objects.library = objects.runtime_library = libraries
    objects.plan_handle = PLAN
    objects._registered = set(ALIASES)
    objects._borrowed = set()
    objects._runtime_object_ids = dict(ALIASES)
    objects._size_by_alias = dict.fromkeys(ALIASES, 4096)

    bridge = RuntimeBridge.__new__(RuntimeBridge)
    bridge.runtime = runtime
    bridge.library = bridge.runtime_library = libraries
    bridge.plan_handle = PLAN
    bridge.objects = objects
    bridge._admitted_task_handles = {1}
    bridge._admitted_acquisitions = {}
    bridge._fixed_layout_installed = True

    def record(name):
        return lambda: libraries.calls.append(name)

    executor = Executor.__new__(Executor)
    executor.runtime = runtime
    executor.device = torch.device("cuda", 0)
    executor._closed = False
    executor._plan_released = False
    executor._failure = failure
    executor._pending = None
    executor._distributed = None
    executor._observations = None
    executor._providers = SimpleNamespace(close=record("providers"))
    executor._completion = SimpleNamespace(release=record("completion"))
    executor._paths = (
        ExecutionPath(
            plan=None,
            memory=PlanMemory(None, None, 0, PLAN),
            bridge=bridge,
            storage=SimpleNamespace(clear_views=record("clear_views")),
            controls=None,
            tasks=(),
            acquisition=0,
        ),
    )
    executor._named_paths = {}
    return executor


def _dispatch_failure(executor, libraries):
    error = NotImplementedError("gradient combination must be part of a compiled task")
    executor._failed(error)
    return error, lambda: None


def _completion_failure(executor, libraries):
    error = AdmissionError("physical budget check failed")

    def finish():
        raise error

    class Outputs:
        pass

    outputs = Outputs()
    alive = weakref.ref(outputs)
    libraries.caller_outputs = lambda: alive() is not None
    executor._pending = InvocationResult(
        outputs, finish, on_resolved=lambda result: None, on_failure=executor._failed
    )
    return error, executor._pending.result


def _latched_failure(executor, libraries):
    error = MemoryError("ShadowSpill out of memory, with nothing left to release")
    executor._failed(error)
    return error, lambda: None


@pytest.mark.parametrize(
    "latched,fail",
    [
        pytest.param(0, _dispatch_failure, id="dispatch"),
        pytest.param(0, _completion_failure, id="completion"),
        pytest.param(Status.NO_PROGRESS, _latched_failure, id="latched-no-progress"),
        # The status a refused close returns, latched instead: the runtime is
        # unpublished, so the close released it rather than refused.
        pytest.param(
            Status.INVALID_STATE, _latched_failure, id="latched-invalid-state"
        ),
    ],
)
def test_one_close_after_a_failure_releases_the_plan_and_the_runtime(
    monkeypatch, synchronized, latched, fail
):
    libraries = _Libraries(latched)
    executor = _executor(libraries, monkeypatch)
    failure, wait = fail(executor, libraries)

    with pytest.raises(BaseException) as raised:
        try:
            wait()
            raise failure
        finally:
            executor.close()

    # close returned, and a latched runtime's refusals left no trace on the
    # error the caller sees.
    assert raised.value is failure
    assert not getattr(failure, "__notes__", ())
    assert "plan_clear_tasks" in libraries.calls and "plan_close" in libraries.calls
    assert any(call.startswith("unregister_object") for call in libraries.calls)
    assert libraries.calls.count("allocator_close") == 1
    assert not executor.runtime._active_plan_handles
    assert executor.runtime._closed and not libraries.published
    assert executor._closed and executor._pending is None
    assert synchronized == [executor.device]
    calls = list(libraries.calls)
    executor.close()
    assert libraries.calls == calls


@pytest.mark.parametrize(
    "latched", [0, Status.NO_PROGRESS], ids=["unlatched", "latched-no-progress"]
)
def test_a_completion_failure_found_by_close_is_raised_once_all_is_released(
    monkeypatch, synchronized, latched
):
    libraries = _Libraries(latched)
    executor = _executor(libraries, monkeypatch)
    failure, _ = _completion_failure(executor, libraries)

    with pytest.raises(AdmissionError) as raised:
        executor.close()

    # Nothing but close waited for the call, so close raises its failure,
    # without the refusals a latched runtime made on the way.
    assert raised.value is failure is executor._failure
    assert failure.__context__ is None
    assert executor.runtime._closed and not executor.runtime._active_plan_handles
    assert synchronized == [executor.device]


def test_a_refused_close_after_a_failure_names_it_and_can_be_retried(
    monkeypatch, synchronized
):
    libraries = _Libraries()
    held = [object()]
    libraries.caller_outputs = lambda: bool(held)
    failure = RuntimeError("dispatch failed")
    executor = _executor(libraries, monkeypatch, failure)

    with pytest.raises(RuntimeConfigurationError, match="caller-owned") as raised:
        executor.close()

    assert raised.value.__cause__ is failure
    assert not executor.runtime._closed and libraries.published
    assert not executor.runtime._active_plan_handles
    held.clear()
    executor.close()
    assert executor._closed and executor.runtime._closed
    assert libraries.calls.count("plan_close") == 1


def test_a_with_block_that_fails_leaves_nothing_held(monkeypatch, synchronized):
    libraries = _Libraries(Status.NO_PROGRESS)
    failure = MemoryError("out of memory")
    executor = _executor(libraries, monkeypatch, failure)

    with pytest.raises(MemoryError) as raised, executor:
        raise failure

    assert raised.value is failure
    assert not getattr(failure, "__notes__", ())
    assert executor.runtime._closed and not executor.runtime._active_plan_handles


def test_a_with_block_notes_a_close_the_runtime_refused(monkeypatch, synchronized):
    libraries = _Libraries()
    libraries.caller_outputs = lambda: True
    failure = MemoryError("out of memory")
    executor = _executor(libraries, monkeypatch, failure)

    with pytest.raises(MemoryError) as raised, executor:
        raise failure

    assert raised.value is failure
    assert any("caller-owned" in note for note in failure.__notes__)
    assert not executor.runtime._closed and not executor.runtime._active_plan_handles


def test_a_device_that_cannot_synchronize_does_not_stop_close(monkeypatch):
    def fail(device):
        raise RuntimeError("illegal memory access")

    monkeypatch.setattr(torch.cuda, "synchronize", fail)
    libraries = _Libraries()
    failure = RuntimeError("kernel failed")
    executor = _executor(libraries, monkeypatch, failure)

    executor.close()

    assert executor._closed and executor.runtime._closed
    assert any("illegal memory access" in note for note in failure.__notes__)


@pytest.mark.cuda
@pytest.mark.fresh_process
def test_one_close_releases_everything_after_an_out_of_memory_failure(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    program = build_forward(nn.Linear(64, 1024), example_args=(torch.randn(1024, 64),))
    save_plan(program, tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            *(["-S"] if sys.flags.no_site else []),
            str(Path(__file__).with_name("_close_after_oom.py")),
            str(tmp_path),
        ],
        env={
            **os.environ,
            "TRITON_CACHE_DIR": str(tmp_path / "empty-triton"),
            "TORCHINDUCTOR_CACHE_DIR": str(tmp_path / "empty-inductor"),
        },
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Closed once after an out-of-memory failure" in result.stdout
    # The exit handler reports a runtime it had to abandon; a closed one has
    # nothing left for it.
    assert "outstanding" not in result.stderr, result.stderr
