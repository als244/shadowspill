"""Trace ownership and failures stay separate from confirmed model effects."""

import hashlib
import io
import json
from dataclasses import replace
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from engine_tests.diagnostics.test_step_file import (
    _ALLOCATION_EVENTS,
    _RUNTIME_EVENTS,
    _with_events,
)
from engine_tests.training.test_worker_loop import SCOPE, call
from shadowspill.diagnostics.collection import DiagnosticAssemblyError
from shadowspill.diagnostics.step_file import StepEvents, read_step_events
from shadowspill.orchestration.launch.publication import OutputFiles, require_output
from shadowspill.pytorch.invocation import InvocationResult
from shadowspill.runtime.plan.tracing import TraceCapacities
from shadowspill.training.worker.execution import WorkerExecution
from shadowspill.training.worker.loop import WorkerLoop
from shadowspill.training.worker.protocol import (
    CODEC,
    WorkerArtifacts,
    WorkerControl,
    WorkerReply,
    ticket,
)
from shadowspill.training.worker.trace_files import TraceBudget, TraceFiles
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.requests import ArtifactRef


def traced_call():
    command = call()
    command = replace(
        command,
        history_id="history",
        completed_updates=4,
        request=replace(command.request, runtime_trace=True),
    )
    return SimpleNamespace(
        command=command,
        plan="main",
        hyperparams={},
        input_loader=lambda _: None,
        verify_inputs=lambda _: None,
        checkpoint=None,
        restore=None,
    )


@pytest.mark.parametrize(
    "fault", [None, "report", "decode", "write", "capacity", "completion"]
)
def test_trace_publication_preserves_completed_effects_and_owned_file_lifetime(
    tmp_path, monkeypatch, fault
):
    disk, memory, scratch = (
        BufferBudget(4 << 20),
        BufferBudget(4 << 20),
        BufferBudget(1 << 20),
    )
    with OutputFiles(root=tmp_path, disk=disk, memory=memory) as files:
        bound = traced_call()
        trace = TraceFiles(
            {
                ("main", False): TraceBudget(
                    TraceCapacities(1024, 1024),
                    1 << 20,
                    1 if fault == "capacity" else 1 << 20,
                    1024 if fault == "report" else 0,
                )
            },
            bundle=ArtifactRef("plans", "a" * 64, 100),
            root=files.root,
        )
        with files.reserve(trace.spec(bound)) as owner:
            trace.check(bound)
            updates = []

            def finish():
                if fault == "completion":
                    raise RuntimeError("completion uncertain")
                updates.append(1)

            def diagnostics():
                if fault == "decode":
                    raise DiagnosticAssemblyError("trace view failed")
                return _with_events()

            def run(**options):
                assert options["runtime_trace"] and options[
                    "trace_capacity"
                ] == TraceCapacities(1024, 1024)
                assert scratch.used == (1 << 20, 1)
                return InvocationResult(
                    object(),
                    finish,
                    on_resolved=lambda _: None,
                    on_failure=lambda _: None,
                    diagnostics=diagnostics,
                )

            executor = SimpleNamespace(run=run, synchronize=lambda: None)
            worker = WorkerExecution(
                executor,
                SimpleNamespace(scope=SCOPE, primary="main"),
                BufferBudget(1),
                lambda *_: None,
                traces=trace,
                diagnostics=scratch,
            )
            worker.prepare(bound)
            if fault == "write":

                def full(*args):
                    raise OSError("trace disk full")

                monkeypatch.setattr(trace, "_write", full)
            owner.begin()
            if fault == "completion":
                with pytest.raises(RuntimeError, match="uncertain"):
                    worker(bound)
                assert not updates
            else:
                result = worker(bound)
                assert isinstance(result, WorkerArtifacts) and updates == [1]
                reply = WorkerReply(
                    ticket(bound.command),
                    "completed",
                    "completed",
                    trace=result.trace,
                    trace_issue=result.trace_issue,
                )
                assert CODEC.decode(CODEC.encode(reply)) == reply
                if fault not in (None, "report"):
                    assert result.trace is None
                    assert result.trace_issue.code == "runtime_trace_failed"
                    assert result.trace_issue.artifact_id == owner.spec.id
                else:
                    assert result.trace is not None and result.trace_issue is None
                    with pytest.raises(RuntimeError, match="not finished"):
                        owner.verify(result.trace)
            assert scratch.used == (0, 0) and disk.used[0] >= owner.spec.size_bytes
            owner.finished()
            if fault in (None, "report"):
                owner.verify(result.trace)
                with owner.open(result.trace) as source:
                    payload = source.read()
                assert hashlib.sha256(payload).hexdigest() == result.trace.sha256
                with ZipFile(io.BytesIO(payload)) as archive:
                    expected = {
                        "manifest.json",
                        "step.json",
                        "step.events.npz",
                    }
                    if fault == "report":
                        expected.add("report_issue.json")
                        assert (
                            "no corresponding saved Plan"
                            in json.loads(archive.read("report_issue.json"))["message"]
                        )
                    assert set(archive.namelist()) == expected
                    manifest = json.loads(archive.read("manifest.json"))
                    assert (
                        manifest["request"] == bound.command.request.identity.to_dict()
                    )
                    assert manifest["completed_updates_before"] == 4
                    assert manifest["history_id"] == "history" and manifest["rank"] == 0
                    assert (
                        manifest["plan"] == "main"
                        and manifest["plans"]["id"] == "plans"
                    )
                    assert json.loads(archive.read("step.json")) == json.loads(
                        json.dumps(_with_events().as_dict(events=False))
                    )
                    path = tmp_path / "downloaded"
                    path.mkdir()
                    for name in archive.namelist():
                        (path / name).write_bytes(archive.read(name))
                assert read_step_events(path / "step.json") == StepEvents(
                    _RUNTIME_EVENTS, _ALLOCATION_EVENTS
                )
                require_output(files.root, owner.spec).write_bytes(b"corrupted")
                # A new verifier catches tampering before a publication borrow.
                owner.reference = None
                with pytest.raises(ValueError, match="size"):
                    owner.verify(result.trace)
    assert disk.used == memory.used == scratch.used == (0, 0)


def test_missing_trace_configuration_rejects_before_execution():
    worker = WorkerExecution(
        SimpleNamespace(),
        SimpleNamespace(scope=SCOPE, primary="main"),
        BufferBudget(1),
        lambda *_: None,
    )
    with pytest.raises(ValueError, match="no admitted trace"):
        worker.prepare(traced_call())


def test_worker_loop_preserves_optional_trace_issue_and_never_replays():
    bound = traced_call()
    effects = []

    def execute(_):
        effects.append(1)
        return WorkerArtifacts(
            None,
            trace=ArtifactRef("trace", "a" * 64, 10, "application/zip"),
            execution_seconds=2.5,
        )

    worker = WorkerLoop(SCOPE, lambda _: bound, execute, BufferBudget(1))
    ready = worker.handle(bound.command)
    completed = worker.handle(WorkerControl(ready.ticket, "execute"))
    assert completed.status == completed.effect == "completed" and worker.valid
    assert completed.trace.id == "trace" and effects == [1]
    assert completed.execution_seconds == 2.5
    assert CODEC.decode(CODEC.encode(completed)).execution_seconds == 2.5
    with pytest.raises(ValueError):
        replace(completed, status="ready", effect="not_started")


@pytest.mark.parametrize("traced", [False, True])
def test_execution_duration_excludes_output_and_trace_report_processing(
    monkeypatch, traced
):
    from shadowspill.training.worker import execution as module

    clock = [100.0]
    monkeypatch.setattr(module, "monotonic", lambda: clock[0])
    call = traced_call()
    call.command = replace(
        call.command, request=replace(call.command.request, runtime_trace=traced)
    )

    def run(**kwargs):
        clock[0] += 2.0
        return SimpleNamespace(result=lambda: None)

    def synchronize():
        clock[0] += 0.5

    def collect(*args):
        clock[0] += 10.0

    def report(*args):
        clock[0] += 20.0
        return None, None

    traces = SimpleNamespace(
        budget=lambda _: (
            SimpleNamespace(host_bytes=1, capacities=None) if traced else None
        ),
        collect=report,
    )
    worker = WorkerExecution(
        SimpleNamespace(run=run, synchronize=synchronize),
        SimpleNamespace(scope=SCOPE, primary="main"),
        BufferBudget(1),
        collect,
        traces=traces,
        diagnostics=BufferBudget(1),
    )
    result = worker(call)
    assert result.execution_seconds == 2.5
    assert clock[0] == (133.0 if traced else 113.0)


@pytest.mark.parametrize("duration", [-1, float("nan"), float("inf"), True])
def test_worker_reply_rejects_invalid_execution_duration(duration):
    with pytest.raises(ValueError):
        WorkerReply(
            ticket(call()), "completed", "completed", execution_seconds=duration
        )
