"""All assigned workers agree before opening mutable execution state."""

import io
import socket
import subprocess
import sys
from contextlib import contextmanager, suppress
from dataclasses import replace
from threading import Barrier, Thread
from types import SimpleNamespace

import pytest

from engine_tests.training.test_preparation_templates import inventory
from engine_tests.training.test_prepared_requests import family
from engine_tests.training.test_worker_group import commands
from engine_tests.training.test_worker_loop import SCOPE, call
from shadowspill.ir import ResourceKind, ResourceSpec, TaskSpec
from shadowspill.training.components import InitializationRuntime
from shadowspill.training.worker.channel import read_message, write_message
from shadowspill.training.worker.loop import WorkerLoop
from shadowspill.training.worker.output_files import FileOutputs
from shadowspill.training.worker.protocol import CODEC, WorkerBoot, WorkerStartup
from shadowspill.training.worker.runtime import PreparedWorker
from shadowspill.training.worker.startup import (
    WorkerStartupConnection,
    await_start,
    report_ready,
    start_workers,
)
from shadowspill_client import Execution
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.codec import ProtocolError, decode
from shadowspill_client.protocol.requests import ArtifactRef

BUNDLE = ArtifactRef("plans", "a" * 64, 128)


def description(rank, **options):
    return WorkerStartup(
        replace(SCOPE, rank=rank), BUNDLE, "b" * 64, (1, 0), 2000, 1000, 500, **options
    )


@contextmanager
def peers(*, different=None, early=None, lost=None):
    endpoints, sockets, threads, events, errors = {}, [], [], [], []
    barrier = Barrier(2)

    def run(rank, child):
        try:
            with child.makefile("rwb", buffering=0) as stream:
                expected = description(rank)
                if rank == different:
                    expected = replace(expected, initialization_sha256="c" * 64)
                if rank == early:
                    report_ready(expected, stream)
                    return
                await_start(expected, stream, stream)
                events.append((rank, "open"))
                # Returning readiness before releasing the other worker would
                # deadlock a real collective initialization at this boundary.
                barrier.wait(timeout=3)
                if rank == lost:
                    return
                report_ready(expected, stream)

                def bind(command):
                    return SimpleNamespace(
                        command=command,
                        checkpoint=None,
                        restore=None,
                        verify_inputs=lambda _: None,
                    )

                def execute(bound):
                    events.append((rank, "execute"))

                WorkerLoop(expected.scope, bind, execute, BufferBudget(1)).serve(
                    stream, stream
                )
        except (ValueError, RuntimeError, OSError) as error:
            errors.append(error)
        finally:
            child.close()

    for rank in range(2):
        parent, child = socket.socketpair()
        parent.settimeout(5)
        child.settimeout(5)
        stream = parent.makefile("rwb", buffering=0)
        endpoints[rank] = WorkerStartupConnection(description(rank), stream, stream)
        sockets.append(parent)
        thread = Thread(target=run, args=(rank, child))
        thread.start()
        threads.append(thread)
    stopped = []

    def stop():
        stopped.append(True)
        barrier.abort()
        for peer in sockets:
            with suppress(OSError):
                peer.shutdown(socket.SHUT_RDWR)
        for thread in threads:
            thread.join(timeout=5)
            assert not thread.is_alive()

    try:
        yield endpoints, stop, events, stopped
    finally:
        for endpoint in endpoints.values():
            endpoint.close()
        if not stopped:
            stop()
        for peer in sockets:
            peer.close()


def test_exact_startup_evidence_is_private_bounded_and_immutable():
    value = WorkerBoot(description(1), "prepared")
    assert CODEC.decode(CODEC.encode(value)) == value
    with pytest.raises(ProtocolError, match="unknown protocol record"):
        decode(CODEC.encode(value))
    with pytest.raises(ValueError, match="startup phase"):
        replace(value, phase="execute")
    for change in (
        {"logical_by_worker": [0, 0]},
        {"logical_by_worker": [True, 0]},
        {"logical_by_worker": []},
        {"scope": replace(SCOPE, rank=2)},
        {"initialization_sha256": "../weights"},
        {"device_bytes": 0},
        {"spill_bytes": -1},
        {"staging_bytes": True},
    ):
        with pytest.raises((TypeError, ValueError)):
            replace(value.startup, **change)


def test_all_cpu_preflights_and_start_releases_precede_collective_readiness():
    with peers() as (endpoints, stop, events, stopped):
        group = start_workers(endpoints, stop=stop)
        try:
            assert sorted(events) == [(0, "open"), (1, "open")]
            assert not stopped
            for endpoint in endpoints.values():
                with pytest.raises(RuntimeError, match="never restart"):
                    endpoint.prepared()
                endpoint.close()  # Ownership has transferred to the group.
            with pytest.raises(RuntimeError, match="never restart"):
                start_workers(endpoints, stop=stop)
            assert not stopped and group.valid
            replies = group.run(commands())
            assert all(reply.status == "completed" for reply in replies.values())
            assert sum(phase == "execute" for _, phase in events) == 2
        finally:
            group.close()
        assert stopped == [True]


@pytest.mark.parametrize("failure", ["different", "early", "lost"])
def test_disagreement_early_ready_and_lost_ready_never_publish_a_live_group(failure):
    with peers(**{failure: 1}) as (endpoints, stop, events, stopped):
        with pytest.raises(ProtocolError, match="evidence"):
            start_workers(endpoints, stop=stop)
        assert stopped == [True]
        assert not any(phase == "execute" for _, phase in events)
        if failure != "lost":
            assert not events  # No worker was released into mutable initialization.


def test_assignment_failure_closes_owned_channels_before_stop():
    streams = [io.BytesIO(), io.BytesIO()]
    endpoints = {
        rank: WorkerStartupConnection(description(rank), stream, stream)
        for rank, stream in enumerate(streams)
    }
    endpoints[1].expected = replace(
        endpoints[1].expected, bundle=replace(BUNDLE, id="other")
    )
    stopped = []

    def stop():
        assert all(stream.closed for stream in streams)
        stopped.append(True)

    with pytest.raises(ValueError, match="disagree"):
        start_workers(endpoints, stop=stop)
    assert stopped == [True]


def test_worker_refuses_foreign_release_before_any_mutable_state():
    incoming, outgoing = io.BytesIO(), io.BytesIO()
    write_message(incoming, WorkerBoot(description(1), "start"))
    incoming.seek(0)
    with pytest.raises(ProtocolError, match="exact admitted startup release"):
        await_start(description(0), incoming, outgoing)


def prepared_worker(inv=None):
    prepared = family(inv)
    # These CPU startup tests use sentinel Plans, so supply synthetic compiler
    # views matching their real prepared output declarations. GPU tests use the
    # actual saved compiler artifacts and independently verify result bytes.
    for rows in prepared.families.values():
        for row in rows:
            for rank, worker in row.workers.items():
                bindings = [
                    binding
                    for (_, logical), binding in sorted(
                        row.source.layout.bindings.items()
                    )
                    if logical == rank
                ]
                for name in worker.members():
                    groups, slots, views = [], [], []
                    for binding in () if name == worker.optimizer else bindings:
                        names = sorted(
                            set(binding.contract.outputs.metrics)
                            | set(binding.tensor_shapes)
                        )
                        group = []
                        for position, output in enumerate(("loss", *names)):
                            if not position and row.operation.operation == "forward":
                                group.append(None)
                                continue
                            leaf = len(views)
                            identity = f"output.{leaf}"
                            group.append(identity)
                            slots.append({"object_id": identity, "leaf_index": leaf})
                            tensor = (
                                binding.contract.outputs.tensors.get(output)
                                if position
                                else None
                            )
                            views.append(
                                {
                                    "leaf_index": leaf,
                                    "dtype": "torch."
                                    + (tensor.dtype if tensor else "float32"),
                                    "shape": binding.tensor_shapes[output]
                                    if tensor
                                    else (),
                                    "layout": "torch.strided",
                                }
                            )
                        groups.append(group)
                    program = prepared.plans[rank][name].program
                    program.tasks = (
                        SimpleNamespace(task_id="task", profile_id="result"),
                    )
                    program.bindings.update(
                        {
                            "kind": "forward"
                            if row.operation.operation == "forward"
                            else "step",
                            "forward_templates": (),
                            "public_outputs": groups,
                            "entrypoints": [{"task_id": "task", "output_slots": slots}],
                            "manifests": {
                                "result": {"storage_contract": {"output_views": views}}
                            },
                        }
                    )
    for plans in prepared.plans.values():
        for plan in plans.values():
            plan.program.bindings["compiled_device_index"] = 0
            # Synthetic trace geometry, matching the existing sentinel Plan.
            # Real trace sizing and codec behavior have separate saved-Plan tests.
            task = TaskSpec(
                "task",
                ResourceSpec("compute", ResourceKind.COMPUTE),
                "result",
                requires_entrypoint=False,
            )
            plan.program.tasks = (task,)
            plan.program.parallelism = None
            plan.selections = ()
            plan._costed_program = SimpleNamespace(
                selected_tasks=lambda _, task=task: (task,), objects=()
            )
            plan.schedule = SimpleNamespace(actions=())
            plan._accumulation_plan = None
            plan.memory_budgets = SimpleNamespace(
                execution_bytes=2 << 30, spill_bytes=200
            )
    prepared = replace(
        prepared,
        memory={
            rank: SimpleNamespace(
                spill_bytes=100, fixed_slice_bytes=128, execution_bytes=256
            )
            for rank in prepared.plans
        },
    )
    return PreparedWorker.resolve(
        prepared, SCOPE, bundle=BUNDLE, rank_map={rank: rank for rank in prepared.plans}
    )


def test_cpu_worker_description_reuses_spill_margin_and_rejects_ordinal_changes():
    prepared = prepared_worker()
    assert prepared.startup.spill_bytes == 105
    assert (
        prepared.startup.staging_bytes == prepared.initialization.description.peak_bytes
    )
    assert prepared.startup.device_bytes == 2 << 30
    for plan in prepared.bindings.prepared.plans[0].values():
        plan.program.bindings["compiled_device_index"] = 1
    with pytest.raises(ValueError, match="compiled for cuda:1"):
        PreparedWorker.resolve(prepared.bindings.prepared, SCOPE, bundle=BUNDLE)


@pytest.mark.parametrize("execution", [Execution(), Execution(reject_overbudget=True)])
@pytest.mark.parametrize("failure", [None, "open", "initialize", "spill", "outputs"])
def test_runtime_owns_one_executor_and_releases_it_after_failure(
    tmp_path, monkeypatch, failure, execution
):
    import torch

    from shadowspill.artifacts.tensors import TensorChunk
    from shadowspill.execution import executor as execution_module
    from shadowspill.pytorch._rng import preserve_rng

    events = []

    class Executor:
        device = torch.device("cpu")
        logical_rank = 0
        spill_capacity_bytes = 104 if failure == "spill" else 105

        def __init__(self, primary, **options):
            events.append("open")
            assert options["reject_overbudget"] is execution.reject_overbudget
            assert options["rank_map"] == {0: 0}
            if failure == "open":
                raise OSError("open failed")

        def initialize(self, fill, *, chunk_bytes):
            events.append("initialize")
            if failure == "initialize":
                raise OSError("initialize failed")
            fill(TensorChunk("head.bias", (3,), torch.float32, 0, 3), torch.empty(3))

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            events.append("close")

    monkeypatch.setattr(execution_module, "Executor", Executor)
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    prepared = prepared_worker(inventory(execution=execution))
    runtime = InitializationRuntime(tmp_path, lambda _: pytest.fail("artifact opened"))
    outputs = FileOutputs(prepared.outputs, root=tmp_path)
    if failure == "outputs":

        def reject(bound):
            events.append("check_output")
            raise OSError("output permit missing")

        monkeypatch.setattr(outputs, "check", reject)
        monkeypatch.setattr(
            prepared.bindings,
            "bind",
            lambda command, resolve: SimpleNamespace(command=command),
        )

    def invoke():
        with (
            preserve_rng(torch.device("cpu")),
            prepared.open(runtime, lambda _: tmp_path, outputs) as running,
        ):
            events.append("ready")
            if failure == "outputs":
                reply = running.loop.handle(call())
                assert reply.status == "failed" and reply.effect == "not_started"
                assert reply.message == "output permit missing"
            assert running.loop.valid
        assert not running.loop.valid

    if failure and failure != "outputs":
        with pytest.raises((OSError, RuntimeError)):
            invoke()
    else:
        invoke()
    assert (
        events
        == {
            None: ["open", "initialize", "ready", "close"],
            "open": ["open"],
            "initialize": ["open", "initialize", "close"],
            "spill": ["open", "initialize", "close"],
            "outputs": ["open", "initialize", "ready", "check_output", "close"],
        }[failure]
    )


@pytest.mark.parametrize("visibility", [None, "", "-1", "0,1"])
def test_worker_requires_explicit_single_device_before_executor_import(
    tmp_path, monkeypatch, visibility
):
    if visibility is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visibility)
    prepared = prepared_worker()
    runtime = InitializationRuntime(tmp_path, lambda _: pytest.fail("artifact opened"))
    with (
        pytest.raises(ValueError, match="explicitly assigned visible device"),
        prepared.open(
            runtime, lambda _: tmp_path, FileOutputs(prepared.outputs, root=tmp_path)
        ),
    ):
        pytest.fail("opened an unassigned worker")


def test_startup_description_imports_without_framework_or_catalog():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'triton', 'shadowspill_workloads'}:
            raise AssertionError('forbidden startup import: ' + fullname)
sys.meta_path.insert(0, Block())
from shadowspill.training.worker.runtime import PreparedWorker
from shadowspill.training.worker.startup import start_workers
from shadowspill.planning.spill import spill_configuration
assert spill_configuration(100, 200, None).capacity == 105
""",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("failure", [None, "release", "initialize", "invoke"])
def test_worker_serve_orders_readiness_and_closes_channels(
    tmp_path, monkeypatch, failure
):
    prepared = prepared_worker()
    events = []

    class Stream(io.BytesIO):
        def close(self):
            if not self.closed:
                self.sent = self.getvalue()
            super().close()

    incoming, outgoing = Stream(), Stream()
    released = prepared.startup
    if failure == "release":
        released = replace(released, initialization_sha256="d" * 64)
    write_message(incoming, WorkerBoot(released, "start"))
    incoming.seek(0)

    @contextmanager
    def open_worker(
        self,
        runtime,
        resolve,
        outputs,
        *,
        checkpoint_stream=None,
        traces=None,
        capture_progress=None,
    ):
        assert capture_progress is None
        events.append("open")
        try:
            if failure == "initialize":
                raise OSError("initialize failed")

            def serve(reader, writer):
                sent = io.BytesIO(writer.getvalue())
                assert read_message(sent) == WorkerBoot(prepared.startup, "prepared")
                assert read_message(sent) == WorkerBoot(prepared.startup, "ready")
                events.append("invoke")
                if failure == "invoke":
                    raise OSError("invocation failed")

            yield SimpleNamespace(loop=SimpleNamespace(serve=serve))
        finally:
            events.append("close")

    monkeypatch.setattr(PreparedWorker, "open", open_worker)

    def run():
        prepared.serve(
            incoming,
            outgoing,
            InitializationRuntime(tmp_path, lambda _: tmp_path),
            lambda _: tmp_path,
            FileOutputs(prepared.outputs, root=tmp_path),
        )

    if failure:
        with pytest.raises((ProtocolError, OSError)):
            run()
    else:
        run()
    assert incoming.closed and outgoing.closed
    if failure == "release":
        assert not events
    else:
        assert events == (
            ["open", "close"]
            if failure == "initialize"
            else ["open", "invoke", "close"]
        )
    sent = io.BytesIO(outgoing.sent)
    assert read_message(sent) == WorkerBoot(prepared.startup, "prepared")
    assert read_message(sent) == (
        None
        if failure in ("release", "initialize")
        else WorkerBoot(prepared.startup, "ready")
    )
