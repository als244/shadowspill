"""Private wire commands resolve only saved forms, views and captured controls."""

import subprocess
import sys
from contextlib import contextmanager
from dataclasses import replace

import pytest

from engine_tests.server.test_request_records import request
from engine_tests.training.test_prepared_requests import family, upload
from shadowspill.training.session.prepared import PreparedRequests
from shadowspill.training.session.state import OperationState
from shadowspill.training.worker.bindings import WorkerBindings, worker_call
from shadowspill.training.worker.protocol import CODEC, WorkerCall, WorkerScope
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.codec import ProtocolError, decode
from shadowspill_client.protocol.payload import snapshot_batch
from shadowspill_client.protocol.requests import OperationRequest

SCOPE = WorkerScope("run", "server", "process-1", 0)


@contextmanager
def invocation(root, operation="step", *, masked=False, accumulated=False):
    from engine_tests.fixtures.definition_operations import records

    prepared = PreparedRequests(family(), cache_root=root)
    datums = records(masked=masked)
    if operation in ("discard", "close"):
        command = OperationRequest(request().identity, operation)
    elif operation == "optim_step":
        command = replace(request(), operation=operation, mode=None)
    else:
        command, _ = upload(
            datums,
            operation=operation,
            outputs="predictions" if operation == "forward" else None,
        )
    state = OperationState()
    if operation == "optim_step" or accumulated:
        state.complete(
            state.begin("forward_backward", "previous", participates=not masked)
        )
    budget = BufferBudget(16 << 20, max_requests=5)
    snapshot = (
        None
        if operation in ("optim_step", "discard", "close")
        else snapshot_batch(datums, budget)
    )
    data = prepared(command, snapshot, budget)
    try:
        transition = state.begin(operation, "a", participates=data.participates)
        wire = worker_call(data, transition, SCOPE, 3)
        files = (
            {}
            if data.spool is None
            else {
                frame.reference.id: frame.path for frame in data.spool.frames.values()
            }
        )
        bindings = WorkerBindings(prepared.prepared, SCOPE)
        yield data, transition, wire, bindings, files
    finally:
        data.close()
        if snapshot is not None:
            snapshot.close()
        assert budget.used == (0, 0)


@pytest.mark.parametrize(
    "operation,masked,accumulated",
    [
        ("step", False, False),
        ("step", True, False),
        ("forward_backward", False, False),
        ("forward_backward", False, True),
        ("forward_backward", True, False),
        ("forward", False, False),
        ("optim_step", False, False),
        ("optim_step", True, False),
        ("discard", False, False),
        ("close", False, False),
    ],
)
def test_wire_rechecks_saved_forms_frames_and_controls(
    tmp_path, operation, masked, accumulated
):
    with invocation(tmp_path, operation, masked=masked, accumulated=accumulated) as (
        data,
        transition,
        command,
        bindings,
        files,
    ):
        restored = CODEC.decode(CODEC.encode(command))
        assert isinstance(restored, WorkerCall) and restored == command
        with pytest.raises(ProtocolError, match="unknown protocol record"):
            decode(CODEC.encode(command))
        bound = bindings.bind(restored, lambda ref: files[ref.id])
        assert bound.plan == data.worker_plans(transition)[0]
        assert bound.controls == data.controls
        budget = BufferBudget(max(1, bound.input_bytes))
        bound.verify_inputs(budget)
        loader = bound.input_loader(budget)
        if operation in ("optim_step", "discard", "close"):
            assert loader is None
        else:
            with loader:
                for i in range(len(bound.frames)):
                    args, kwargs = loader(i)
                    assert kwargs == {}
                    assert {
                        name: value.item() for name, value in args[2].items()
                    } == dict(data.controls.loss_params)
                    del args, kwargs
        assert budget.used == (0, 0)
        assert bool(bound.hyperparams) == (
            operation in ("step", "optim_step") and not masked
        )
        if bound.hyperparams:
            assert bound.hyperparams["scale"] == 1
        assert bound.command.accumulate_gradients == accumulated


@pytest.mark.parametrize(
    "change",
    [
        {"candidate": "saved-plan"},
        {"skipped": True},
        {"accumulate_gradients": True},
    ],
)
def test_discard_cannot_select_compute_forms(tmp_path, change):
    with (
        invocation(tmp_path, "discard") as (_, _, command, _, _),
        pytest.raises(ValueError, match="no compute candidate"),
    ):
        replace(command, **change)


def test_commands_cannot_cross_scope_candidate_or_input_abi(tmp_path):
    with invocation(tmp_path) as (_, _, command, bindings, files):

        def resolve(ref):
            return files[ref.id]

        for patch, match in (
            (
                {"scope": replace(SCOPE, worker_incarnation="process-2")},
                "another worker",
            ),
            ({"scope": replace(SCOPE, rank=1)}, "another worker"),
            ({"candidate": "missing"}, "no saved candidate"),
            ({"frames": command.frames[:-1]}, "complete saved schedule"),
            (
                {
                    "request": replace(
                        command.request,
                        operation="forward",
                        optim={},
                        grad_scale=None,
                        outputs="predictions",
                        mode="evaluation",
                    )
                },
                "candidate differs",
            ),
            (
                {
                    "frames": (
                        replace(command.frames[0], size_bytes=1),
                        *command.frames[1:],
                    )
                },
                "prepared frame",
            ),
            (
                {"request": replace(command.request, optim={"lr": 1e100})},
                "captured scalar",
            ),
        ):
            with pytest.raises(ValueError, match=match):
                bindings.bind(replace(command, **patch), resolve)
        with pytest.raises(ValueError, match="scope differs"):
            replace(command, scope=replace(SCOPE, run_id="different"))
        with pytest.raises(ValueError, match="opaque identifier"):
            replace(command, candidate="../../plan")
        with pytest.raises(ValueError, match="distinct"):
            replace(command, frames=(command.frames[0], command.frames[0]))
        with pytest.raises(ValueError, match="participating backward"):
            replace(command, accumulate_gradients=True)


def test_complete_local_integrity_check_rejects_corruption_before_execution(tmp_path):
    with invocation(tmp_path) as (_, _, command, bindings, files):
        bound = bindings.bind(command, lambda ref: files[ref.id])
        budget = BufferBudget(bound.input_bytes)
        path = files[command.frames[-1].id]
        with path.open("r+b") as stream:
            stream.seek(-1, 2)
            previous = stream.read(1)[0]
            stream.seek(-1, 2)
            stream.write(bytes([previous ^ 1]))
        with pytest.raises(ProtocolError, match="checksum"):
            bound.verify_inputs(budget)
        assert budget.used == (0, 0)


def test_worker_binding_and_protocol_imports_do_not_load_gpu_or_catalog_modules():
    code = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'triton', 'shadowspill_workloads'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from shadowspill.training.worker.bindings import WorkerBindings
from shadowspill.training.worker.loop import WorkerLoop
from shadowspill.training.worker.execution import WorkerExecution
from shadowspill.training.worker.protocol import CODEC
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
