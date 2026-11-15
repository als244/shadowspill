"""Service lifetime over real CPU children and the existing ordered controller."""

import hashlib
import io
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import pytest

from engine_tests.server.test_storage import IDENTITY, storage
from engine_tests.training.test_worker_startup import prepared_worker
from shadowspill.orchestration.launch.admission import (
    AssignedDevice,
    NodeAdmission,
    NodeCapacity,
    ReservationCancelled,
    ResourceRequest,
)
from shadowspill.orchestration.launch.host_arena import HostArena
from shadowspill.server.live import LiveRun
from shadowspill.training.worker import launch as launch_module
from shadowspill.training.worker.node import WorkerNode
from shadowspill.training.worker.resources import WorkerHost
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.codec import encode
from shadowspill_client.protocol.requests import (
    ArtifactRef,
    OperationRequest,
    RequestIdentity,
)

CHILD = """
import socket, sys
from types import SimpleNamespace
from shadowspill.training.worker.channel import read_message
from shadowspill.training.worker.startup import await_start, report_ready
from shadowspill.training.worker.loop import WorkerLoop
from shadowspill_client.buffers import BufferBudget
with socket.socket(fileno=int(sys.argv[1])) as peer:
    with peer.makefile('rwb', buffering=0) as stream:
        launch = read_message(stream)
        await_start(launch.startup, stream, stream)
        report_ready(launch.startup, stream)
        def bind(command):
            return SimpleNamespace(
                command=command, checkpoint=None, verify_inputs=lambda _: None
            )
        loop = WorkerLoop(launch.startup.scope, bind, lambda _: None, BufferBudget(1))
        loop.serve(stream, stream)
print('service-owned CPU worker exited')
"""


@contextmanager
def configured(tmp_path, monkeypatch, *, register=True):
    worker = prepared_worker()
    family = worker.bindings.prepared
    payload = b"CPU fixture startup bundle, no model execution"
    plans = ArtifactRef("plans", hashlib.sha256(payload).hexdigest(), len(payload))
    worker = replace(worker, startup=replace(worker.startup, bundle=plans))
    admission = NodeAdmission(
        NodeCapacity("local", (AssignedDevice("gpu", "0", 4 << 30),), 8 << 30)
    )
    disk = BufferBudget(8 << 20, 32)
    config = encode(family.inventory.layouts.setup.config)
    config_ref = ArtifactRef("config", hashlib.sha256(config).hexdigest(), len(config))
    monkeypatch.setattr(
        launch_module,
        "worker_command",
        lambda module, arguments: [sys.executable, "-c", CHILD, str(arguments[1])],
    )
    execution = family.inventory.layouts.setup.config.execution
    with (
        HostArena(admission, 64 << 20, 100) as transfers,
        HostArena(admission, 16 << 20, 2) as commands,
        HostArena(admission, 64 << 20, 3) as publication,
        HostArena(
            admission, execution.max_staging_bytes, execution.max_pending_requests
        ) as uploads,
        storage(tmp_path) as service,
    ):
        service.recover()
        with service.artifacts("owner") as artifacts:
            for ref, body in ((config_ref, config), (plans, payload)):
                artifacts.put(ref, lambda _, body=body: io.BytesIO(body))
        if register:
            service.catalog.create(
                IDENTITY, request_sha256=config_ref.sha256, configuration=config_ref
            )
        node = WorkerNode(
            admission,
            {0: worker},
            {0: "gpu"},
            {0: WorkerHost(1 << 20, 1 << 20)},
            packages=(),
            root=tmp_path,
            cache=tmp_path,
            disk=disk,
            memory=transfers.budget,
            environment=os.environ,
            log_limit_bytes=1024,
        )
        live = LiveRun(
            service,
            IDENTITY,
            family,
            [node],
            plans=plans,
            history_id="history",
            root=tmp_path,
            metadata=commands.budget,
            uploads=uploads.budget,
            publication=publication.budget,
            disk=disk,
        )
        try:
            yield SimpleNamespace(
                live=live,
                node=node,
                admission=admission,
                storage=service,
                disk=disk,
                configuration=config_ref,
            )
        finally:
            live.stop()
            assert live.closed
    assert admission.used.claims == 0 and disk.used == (0, 0)


def test_real_startup_ordered_close_and_persistent_rank_logs(tmp_path, monkeypatch):
    with configured(tmp_path, monkeypatch) as ctx:
        assert ctx.storage.catalog.get(IDENTITY).state == "preparing"
        with pytest.raises(RuntimeError, match="initialized"):
            _ = ctx.live.controller
        ctx.live.start()
        info = ctx.storage.catalog.get(IDENTITY)
        assert info.state == "ready" and info.initial_history_id == "history"
        controller = ctx.live.controller
        lease = controller.attach("writer").lease
        command = OperationRequest(
            RequestIdentity(
                "close", "run", 1, lease.writer_id, lease.writer_epoch, "server"
            ),
            "close",
        )
        controller.register(command)
        controller.receive("close")
        assert controller.wait("close", timeout=5).state == "completed"
        controller.drain(timeout=5)
        ctx.live.close()
        assert ctx.live.closed and ctx.node.closed and ctx.node.lease.closed
        assert ctx.storage.catalog.get(IDENTITY).state == "closed"
        assert ctx.storage._requests.used[1] == ctx.storage._artifacts.used[1] == 0
        with ctx.storage.requests(IDENTITY) as journal:
            assert journal.get("close")[1].result.operation == "close"
        with ctx.storage.artifacts("owner") as artifacts:
            manifest = next(
                ref for ref in artifacts.page() if ref.id.startswith("worker-logs.")
            )
            with artifacts.open(manifest) as stream:
                metadata = json.load(stream)
            assert metadata["session_id"] == "session" and metadata["rank"] == 0
            ref = ArtifactRef.from_dict(metadata["streams"]["stdout"])
            with artifacts.open(ref) as stream:
                assert stream.read() == b"service-owned CPU worker exited\n"
        assert ctx.live._family is None and ctx.live._controller is None


def test_stop_retains_partial_start_until_the_waiting_thread_exits(
    tmp_path, monkeypatch
):
    with configured(tmp_path, monkeypatch) as ctx:
        occupied = ctx.admission.reserve(ResourceRequest({"gpu": 4 << 30}, 0))
        entered = Event()
        reserve = ctx.admission.reserve

        def waiting(*args, **kwargs):
            entered.set()
            return reserve(*args, **kwargs)

        monkeypatch.setattr(ctx.admission, "reserve", waiting)
        try:
            with ThreadPoolExecutor(1) as pool:
                future = pool.submit(ctx.live.start, wait_for_resources=True)
                assert entered.wait(5)
                with pytest.raises(ExceptionGroup, match="confirmed exit"):
                    ctx.live.stop()
                assert ctx.storage._requests.used[1] == 1 and not ctx.live.closed
                with pytest.raises(ReservationCancelled):
                    future.result(timeout=5)
            ctx.live.stop()
            assert ctx.live.closed and not occupied.closed
            assert ctx.storage.catalog.get(IDENTITY).state == "preparing"
        finally:
            occupied.close()


@pytest.mark.parametrize("stage", ["publication", "files", "catalog"])
def test_failed_retirement_keeps_storage_and_retries_without_restarting(
    tmp_path, monkeypatch, stage
):
    with configured(tmp_path, monkeypatch) as ctx:
        ctx.live.start()
        controller = ctx.live.controller
        target, name = {
            "publication": (ctx.live, "_publish_logs"),
            "files": (ctx.node, "close"),
            "catalog": (ctx.storage.catalog, "finish"),
        }[stage]
        original = getattr(target, name)

        def failed(*args):
            raise OSError("retirement fault")

        monkeypatch.setattr(target, name, failed)
        with pytest.raises((OSError, ExceptionGroup)):
            ctx.live.close()
        assert controller.closed and ctx.node.lease.closed and not ctx.live.closed
        assert ctx.storage._requests.used[1] == ctx.storage._artifacts.used[1] == 1
        with pytest.raises(RuntimeError, match="closing"):
            ctx.live.start()
        monkeypatch.setattr(target, name, original)
        ctx.live.close()
        assert ctx.live.closed and ctx.storage.catalog.get(IDENTITY).state == "closed"


def test_allocation_stop_of_ready_idle_run_records_loss_not_an_orderly_close(
    tmp_path, monkeypatch
):
    with configured(tmp_path, monkeypatch) as ctx:
        ctx.live.start()
        controller = ctx.live.controller
        ctx.live.stop()
        assert controller.closed and not controller.boundary.valid
        assert ctx.storage.catalog.get(IDENTITY).state == "stopped"
        assert ctx.storage.catalog.get(IDENTITY).error.code == "allocation_stopped"


def test_missing_bundle_cleans_pins_without_starting_a_child(tmp_path, monkeypatch):
    with configured(tmp_path, monkeypatch) as ctx:
        ctx.live.plans = replace(ctx.live.plans, id="missing")
        with pytest.raises(ValueError, match="not published"):
            ctx.live.start()
        ctx.live.close()
        assert ctx.node.closed and not ctx.node.launched and ctx.live.closed
        assert ctx.storage._requests.used[1] == ctx.storage._artifacts.used[1] == 0


def test_partial_launch_failure_retains_empty_log_reservations_until_cleanup(
    tmp_path, monkeypatch
):
    from shadowspill.training.worker import node as node_module

    with configured(tmp_path, monkeypatch) as ctx:

        def failed(*args, **kwargs):
            raise OSError("child launch failed")

        monkeypatch.setattr(node_module, "launch_worker", failed)
        with pytest.raises(OSError, match="child launch"):
            ctx.live.start()
        assert ctx.node.logs and ctx.disk.used[0] > 0
        ctx.live.close()
        assert ctx.live.closed and ctx.node.closed and ctx.disk.used == (0, 0)


def test_failure_committing_readiness_keeps_workers_owned_for_stop(
    tmp_path, monkeypatch
):
    with configured(tmp_path, monkeypatch) as ctx:

        def failed(*args, **kwargs):
            raise OSError("readiness commit failed")

        monkeypatch.setattr(ctx.storage.catalog, "ready", failed)
        with pytest.raises(OSError, match="readiness commit"):
            ctx.live.start()
        assert ctx.live._controller is not None and ctx.node.launched
        assert not ctx.live.ready and not ctx.node.lease.closed
        ctx.live.stop()
        assert ctx.live.closed and ctx.node.lease.closed
        assert ctx.storage.catalog.get(IDENTITY).state == "preparing"
