"""Public service composition owns actual HTTP and CPU sentinel worker lifetimes."""

import os
import socket
import subprocess
import sys
from contextlib import contextmanager

import pytest

from engine_tests.server.test_http import TOKEN
from engine_tests.server.test_jobs import until
from engine_tests.server.test_local_creation import configured
from engine_tests.training.test_setup_resolution import arguments
from shadowspill.orchestration.launch.admission import (
    AssignedDevice,
    NodeAdmission,
    NodeCapacity,
)
from shadowspill.server import Allocation, Server
from shadowspill.server import allocation as allocation_module
from shadowspill.server.authentication import BearerTokens
from shadowspill_client import ServiceClient


@contextmanager
def deployment(tmp_path, monkeypatch):
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    with configured(fixture, monkeypatch) as ctx:
        nodes = []

        def local(self, root, *, controller_host_bytes):
            assert "SHADOWSPILL_TOKEN" not in self.environment
            node = NodeAdmission(
                NodeCapacity(
                    "cpu-server", (AssignedDevice("gpu", "0", 4 << 30),), 16 << 30
                )
            )
            nodes.append(node)
            return node

        monkeypatch.setattr(Allocation, "_local_node", local)
        server = Server(
            allocation=Allocation.from_environment(),
            service_root=tmp_path / "service",
            authentication=BearerTokens({TOKEN: "owner"}),
        )
        assert server._node is None and not (tmp_path / "service").exists()
        try:
            yield server, ctx, nodes
        finally:
            server.close()
        assert all(node.used.claims == 0 for node in nodes)


def connect(server, **kwargs):
    host, port = server.address
    return ServiceClient(f"http://{host}:{port}", token=TOKEN, **kwargs)


def test_public_server_creates_workers_and_recovers_read_only_records_after_reopen(
    tmp_path, monkeypatch
):
    with deployment(tmp_path, monkeypatch) as (server, ctx, nodes):
        server.start(port=0)
        with connect(server) as service:
            assert (
                service.describe_training(**arguments())
                .result(5)
                .config.model.definition
            )
            training = service.create_training_client(config=ctx.config).result(10)
            until(lambda: server._jobs.used == (1, 0))
            assert server._jobs.used == (1, 0) and nodes[0].used.devices
            summary = training.plans.summary().result(5)
            assert summary.plans == training.plans.reference
            assert summary.candidate_count and summary.plan_count
            # Preview has its own bounded CPU envelope while this run is live.
            assert (
                service.describe_training(config=ctx.config).result(5).config
                == ctx.config
            )
            closed = training.close().result(5)
            session_id, run_id = training.session_id, training.run_id
        server.close()
        assert server.closed and nodes[0].used.claims == 0
        with Server(
            allocation=Allocation.from_environment(),
            service_root=server.root,
            authentication=BearerTokens({TOKEN: "owner"}),
        ) as reopened:
            reopened.start(port=0)
            with connect(reopened, session_id=session_id) as service:
                readonly = service.get_training_client(run_id, read_only=True).result(5)
                assert readonly.plans.summary().result(5) == summary
                assert (
                    readonly.requests.get(closed.identity.request_id).result(5).record
                    == closed.record
                )
            assert not nodes[1].used.devices and reopened._jobs.used == (0, 0)


def test_failed_storage_retirement_keeps_node_claims_and_retries(tmp_path, monkeypatch):
    with deployment(tmp_path, monkeypatch) as (server, _, nodes):
        server.start(port=0)
        storage = server._storage
        original = storage.close

        def fail():
            raise OSError("fixture retained storage cleanup")

        monkeypatch.setattr(storage, "close", fail)
        with pytest.raises(OSError, match="retained storage"):
            server.close()
        assert not server.closed and server._storage is storage
        assert nodes[0].used.host_bytes == server.limits.fixed_host_bytes
        assert not server._host.closed
        monkeypatch.setattr(storage, "close", original)
        server.close()
        assert server.closed and nodes[0].used.claims == 0


def test_failed_listener_start_releases_every_constructed_owner(tmp_path, monkeypatch):
    with (
        deployment(tmp_path, monkeypatch) as (server, _, nodes),
        socket.socket() as occupied,
    ):
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        with pytest.raises(OSError):
            server.start(port=occupied.getsockname()[1])
        assert server.closed and nodes[0].used.claims == 0
        assert server._jobs is server._storage is server._http is None


def test_allocation_snapshots_assignments_and_refuses_partial_multinode_deployment(
    tmp_path, monkeypatch
):
    values = {"CUDA_VISIBLE_DEVICES": "2,1", "SHADOWSPILL_TOKEN": TOKEN}
    allocation = Allocation(values)
    values["CUDA_VISIBLE_DEVICES"] = "0"
    assert allocation.environment["CUDA_VISIBLE_DEVICES"] == "2,1"
    assert "SHADOWSPILL_TOKEN" not in allocation.environment and TOKEN not in repr(
        allocation
    )
    monkeypatch.setattr(
        allocation_module,
        "discover_capacity",
        lambda **kwargs: pytest.fail("probed an unsupported deployment"),
    )
    with pytest.raises(NotImplementedError, match="allocation-agent bootstrap"):
        Allocation({"SLURM_JOB_NUM_NODES": "2"})._local_node(
            tmp_path, controller_host_bytes=1
        )


def test_record_imports_do_not_load_public_service_or_client_runtime():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from shadowspill_client import Model, Datum
from shadowspill.server.records.requests import RequestRecords
assert 'shadowspill_client.service' not in sys.modules
assert 'shadowspill_client.training' not in sys.modules
assert 'shadowspill_client.transport.http' not in sys.modules
assert 'shadowspill.server.service' not in sys.modules
from shadowspill_client import ServiceClient, TrainingClient, SSH
from shadowspill.server import Server, Allocation
assert ServiceClient is __import__('shadowspill_client').ServiceClient
assert Server is __import__('shadowspill.server', fromlist=['Server']).Server
assert not any(name in sys.modules for name in ('torch', 'triton', 'pynvml'))
""",
        ],
        text=True,
        capture_output=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_blocking_serve_sigint_waits_for_listener_exit_and_releases_owners(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import os
import signal
import sys
from pathlib import Path
from threading import Event, Timer
from shadowspill.server import Allocation, Server
from shadowspill.server.authentication import BearerTokens
from shadowspill.server.http import HTTPService
from shadowspill.orchestration.launch.admission import (
    AssignedDevice, NodeAdmission, NodeCapacity,
)
Allocation._local_node = lambda *args, **kwargs: NodeAdmission(
    NodeCapacity('cpu', (AssignedDevice('gpu', '0', 4 << 30),), 16 << 30)
)
finished = Event()
original = HTTPService.serve_forever
def listen(self, **kwargs):
    try:
        original(self, **kwargs)
    finally:
        finished.set()
HTTPService.serve_forever = listen
server = Server(
    allocation=Allocation({}), service_root=Path(sys.argv[1]) / 'service',
    authentication=BearerTokens({'test-' + '1' * 64: 'owner'}),
)
start = server.start
def interrupt_after_start(*args, **kwargs):
    result = start(*args, **kwargs)
    Timer(0.1, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
    return result
server.start = interrupt_after_start
try:
    server.serve(port=0)
except KeyboardInterrupt:
    pass
else:
    raise AssertionError('did not receive SIGINT')
assert server.closed and finished.wait(1)
assert server._node.used.claims == 0
""",
            str(tmp_path),
        ],
        text=True,
        capture_output=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
