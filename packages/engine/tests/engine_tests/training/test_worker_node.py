"""Production node ownership around real private-channel CPU child processes."""

import hashlib
import io
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event

import pytest

from engine_tests.training.test_worker_startup import prepared_worker
from shadowspill.orchestration.capture.files import CollectorLimits
from shadowspill.orchestration.capture.nsys import NsysCapture
from shadowspill.orchestration.launch.admission import (
    AssignedDevice,
    NodeAdmission,
    NodeCapacity,
    ReservationCancelled,
    ResourceRequest,
)
from shadowspill.orchestration.launch.host_arena import HostArena
from shadowspill.orchestration.launch.publication import OutputSpec
from shadowspill.orchestration.launch.staging import FileDelivery
from shadowspill.training.worker import launch as launch_module
from shadowspill.training.worker import node as node_module
from shadowspill.training.worker.launch import start_processes
from shadowspill.training.worker.node import WorkerNode
from shadowspill.training.worker.resources import WorkerHost
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.codec import ProtocolError
from shadowspill_client.protocol.requests import ArtifactRef

PAYLOAD = b"opaque startup artifact"
REFERENCE = ArtifactRef("bundle", hashlib.sha256(PAYLOAD).hexdigest(), len(PAYLOAD))
CHILD = """
import socket, sys
from shadowspill.training.worker.channel import read_message
from shadowspill.training.worker.protocol import WorkerLaunch
from shadowspill.training.worker.startup import await_start, report_ready
with socket.socket(fileno=int(sys.argv[1])) as peer:
    with peer.makefile('rwb', buffering=0) as stream:
        launch = read_message(stream)
        assert isinstance(launch, WorkerLaunch)
        await_start(launch.startup, stream, stream)
        report_ready(launch.startup, stream)
        while stream.read(1):
            pass
print('owned CPU worker stopped')
"""


def test_stop_interrupts_only_its_pending_resource_wait(tmp_path, monkeypatch):
    node, admission, arena, disk = configured(tmp_path)
    occupied = admission.reserve(ResourceRequest({"a": 100}, 0))
    waiting = Event()
    original = admission.reserve

    def reserve(*args, **kwargs):
        waiting.set()
        return original(*args, **kwargs)

    monkeypatch.setattr(admission, "reserve", reserve)
    try:
        with ThreadPoolExecutor(1) as pool:
            pending = pool.submit(node.start, lambda _: io.BytesIO(PAYLOAD), wait=True)
            assert waiting.wait(5)
            with pytest.raises(RuntimeError, match="creation has not finished"):
                node.stop()
            with pytest.raises(ReservationCancelled):
                pending.result(timeout=5)
        node.close()
        assert not occupied.closed
        assert arena.budget.used == disk.used == (0, 0)
        # Stopping this job did not close admission or release another job's GPU.
        with admission.reserve(ResourceRequest({"b": 100}, 0)):
            pass
    finally:
        node.close()
        occupied.close()
        arena.close()
    assert admission.used.claims == 0


def test_stop_during_startup_copy_retains_storage_until_producer_settles(tmp_path):
    node, admission, arena, disk = configured(tmp_path)
    copying, release = Event(), Event()

    class Source(io.BytesIO):
        def read(self, size):
            copying.set()
            assert release.wait(5)
            return super().read(size)

    try:
        with ThreadPoolExecutor(1) as pool:
            pending = pool.submit(node.start, lambda _: Source(PAYLOAD))
            assert copying.wait(5)
            with pytest.raises(RuntimeError, match="creation has not finished"):
                node.close()
            assert node.lease.closed and not node.closed
            assert disk.used[0] > 0 and arena.budget.used[0] > 0
            assert admission.used.claims == 1
            release.set()
            with pytest.raises(InterruptedError, match="cancelled"):
                pending.result(timeout=5)
        node.close()
        assert disk.used == arena.budget.used == (0, 0)
    finally:
        release.set()
        node.close()
        arena.close()


def configured(
    tmp_path, *, disk_bytes=16 << 20, capture_limits=None, initialization_disk_bytes=0
):
    original = prepared_worker()
    original = replace(
        original,
        initialization=replace(
            original.initialization,
            description=replace(
                original.initialization.description,
                disk_bytes=initialization_disk_bytes,
            ),
        ),
    )
    if capture_limits is not None:
        setup = original.bindings.setup
        original.bindings.setup = replace(
            setup,
            config=replace(
                setup.config,
                execution=replace(setup.config.execution, nsys={"gpu_metrics": False}),
            ),
        )
    workers = {
        rank: replace(
            original,
            startup=replace(
                original.startup,
                scope=replace(
                    original.startup.scope,
                    rank=rank,
                    worker_incarnation=f"worker-{rank}",
                ),
                logical_by_worker=(1, 0),
                bundle=REFERENCE,
                device_bytes=100,
            ),
        )
        for rank in (0, 1)
    }
    admission = NodeAdmission(
        NodeCapacity(
            "local",
            (AssignedDevice("a", "0", 100), AssignedDevice("b", "1", 100)),
            8 << 30,
        )
    )
    arena, disk = HostArena(admission, 64 << 20, 100), BufferBudget(disk_bytes, 100)
    node = WorkerNode(
        admission,
        workers,
        {0: "a", 1: "b"},
        {rank: WorkerHost(1 << 20, 1 << 20) for rank in workers},
        packages=(),
        root=tmp_path,
        cache=tmp_path,
        disk=disk,
        memory=arena.budget,
        environment={**os.environ, "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "12345"},
        log_limit_bytes=1024,
        capture_limits=capture_limits,
    )
    return node, admission, arena, disk


def cpu_workers(monkeypatch):
    def command(module, arguments):
        assert module == "shadowspill.training.worker.bootstrap"
        assert arguments[0] == "--control-fd"
        return [sys.executable, "-c", CHILD, str(arguments[1])]

    monkeypatch.setattr(launch_module, "worker_command", command)


def test_initialization_disk_is_reserved_before_sources_or_children(
    tmp_path, monkeypatch
):
    node, admission, arena, disk = configured(
        tmp_path, disk_bytes=16 << 20, initialization_disk_bytes=9 << 20
    )
    monkeypatch.setattr(
        node_module, "launch_worker", lambda *a, **k: pytest.fail("launched a child")
    )
    try:
        with pytest.raises(ValueError, match="staging bytes"):
            node.start(lambda _: pytest.fail("opened a source"))
    finally:
        node.close()
        arena.close()
    assert admission.used.claims == 0
    assert disk.used == (0, 0)
    assert not list(tmp_path.glob("initialization-*"))


@pytest.mark.parametrize("fail", [False, True])
def test_initialization_cache_is_private_reserved_and_removed_after_stop(
    tmp_path, monkeypatch, fail
):
    cpu_workers(monkeypatch)
    node, admission, arena, disk = configured(
        tmp_path, initialization_disk_bytes=1 << 20
    )
    existing = tmp_path / "existing-user-cache"
    existing.write_bytes(b"preserve")
    caches = []
    original_launch = node_module.launch_worker

    def launch(*args, **kwargs):
        cache = kwargs["cache"]
        caches.append(cache)
        assert cache.parent == tmp_path and cache != tmp_path
        assert disk.used[0] >= 2 << 20
        (cache / "download").write_bytes(b"owned initialization bytes")
        if fail:
            raise RuntimeError("initialization startup failure")
        return original_launch(*args, **kwargs)

    monkeypatch.setattr(node_module, "launch_worker", launch)
    try:
        if fail:
            with pytest.raises(RuntimeError, match="startup failure"):
                node.start(lambda _: io.BytesIO(PAYLOAD), startup_timeout=5)
        else:
            processes = node.start(lambda _: io.BytesIO(PAYLOAD), startup_timeout=5)
            with start_processes(processes, stop=node.stop):
                assert len(caches) == 2 and caches[0] == caches[1]
        assert caches[0].is_dir()  # Keep the disk claim until node cleanup succeeds.
        assert disk.used[0] >= 2 << 20
    finally:
        node.close()
        arena.close()
    assert not caches[0].exists()
    assert existing.read_bytes() == b"preserve"
    assert admission.used.claims == 0 and disk.used == (0, 0)


def collector_workers(monkeypatch, child=CHILD, *, fault=None):
    """CPU sentinel: real child/collector exits, with fabricated capture evidence."""
    collector = (
        f"fault = {fault!r}\n"
        + """
import os, subprocess, sys, time
from pathlib import Path
worker = sys.argv[2:]
descriptor = int(worker[-2])
assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) == 1
process = subprocess.Popen(worker, pass_fds=(descriptor,))
os.close(descriptor)
status = process.wait()
time.sleep(.1)  # Final report cannot exist until the worker has exited.
if fault != 'missing':
    Path(sys.argv[1]).write_bytes(b'CPU sentinel report')
raise SystemExit(17 if fault == 'collector' else status)
"""
    )
    progress = (
        f"fault = {fault!r}\n"
        + """
import os
from pathlib import Path
from shadowspill.orchestration.capture.progress import write_progress
from shadowspill_client.protocol.captures import NsightWindow
path = Path(sys.argv[2])
assert path.parent == Path(os.environ['TMPDIR']).parent
state = 'running' if fault == 'partial' else 'complete'
write_progress(path, NsightWindow(2, 2, state, 1 if fault == 'partial' else 2,
                                 'history', 'first', 'last'))
"""
    )

    def command(module, arguments):
        assert module == "shadowspill.training.worker.bootstrap"
        assert arguments[-2] == "--nsys-progress"
        return [
            sys.executable,
            "-c",
            child + progress,
            str(arguments[1]),
            str(arguments[-1]),
        ]

    def capture(self, worker, *, output):
        return [sys.executable, "-c", collector, str(output) + ".nsys-rep", *worker]

    monkeypatch.setattr(launch_module, "worker_command", command)
    monkeypatch.setattr(NsysCapture, "command", capture)


def test_collectors_reserve_before_spawn_and_finalize_after_worker_exit(
    tmp_path, monkeypatch
):
    collector_workers(monkeypatch)
    limits = CollectorLimits(
        host_bytes=1 << 20,
        disk_bytes=128 << 10,
        max_files=16,
        finish_seconds=2,
        executable="/bin/true",
    )
    node, admission, arena, disk = configured(tmp_path, capture_limits=limits)
    base = sum(
        WorkerHost(1 << 20, 1 << 20).peak_bytes(worker.startup)
        for worker in node.prepared.values()
    )
    assert node._request.host_bytes == base + 2 * limits.host_bytes
    original_launch = node_module.launch_worker

    def launch(*args, **kwargs):
        assert set(node.captures) == {0, 1}
        assert disk.used[0] >= 4 * limits.disk_bytes
        return original_launch(*args, **kwargs)

    monkeypatch.setattr(node_module, "launch_worker", launch)
    try:
        processes = node.start(lambda _: io.BytesIO(PAYLOAD), startup_timeout=5)
        with start_processes(processes, stop=node.stop):
            for rank, capture in node.captures.items():
                with pytest.raises(RuntimeError, match="termination"):
                    capture.outcome()
                with pytest.raises(ValueError, match="another"):
                    capture.check(node.lease, node.devices[1 - rank])
        assert all(p.child.process.returncode == 0 for p in processes.values())
        assert admission.used.claims == 1
        for capture in node.captures.values():
            outcome = capture.outcome()
            assert outcome.status == "ready" and outcome.window.completed_updates == 2
            with capture.open(outcome.report) as stream:
                assert stream.read() == b"CPU sentinel report"
        node.close()
        assert disk.used == arena.budget.used == (0, 0)
        assert not list(tmp_path.iterdir())
    finally:
        node.close()
        arena.close()


def test_all_startup_files_and_logs_are_reserved_before_any_child(
    tmp_path, monkeypatch
):
    cpu_workers(monkeypatch)
    node, admission, arena, disk = configured(tmp_path)
    assert admission.used.claims == 1  # Only the caller's host arena so far.
    assert not list(tmp_path.iterdir())
    reads, launches = [], []
    original_launch = node_module.launch_worker

    def launch(*args, **kwargs):
        assert set(node.logs) == {0, 1}
        assert all(log.directory.parent.exists() for log in node.logs.values())
        assert disk.used[0] >= len(PAYLOAD) + 2 * (2048 + 12288)
        assert reads == [REFERENCE]
        launches.append(args[2].startup.scope.rank)
        return original_launch(*args, **kwargs)

    def source(ref):
        reads.append(ref)
        return io.BytesIO(PAYLOAD)

    monkeypatch.setattr(node_module, "launch_worker", launch)
    try:
        processes = node.start(source, startup_timeout=5)
        assert launches == [0, 1]
        with pytest.raises(RuntimeError, match="once"):
            node.start(source)
        files = node.files
        with start_processes(processes, stop=node.stop) as group:
            assert group.valid and admission.used.claims == 2
        assert all(process.child.closed for process in processes.values())
        assert admission.used.claims == 1 and files.grants.root.exists()
        for logs in node.logs.values():
            with logs.open(logs.references()["stdout"]) as stream:
                assert stream.read() == b"owned CPU worker stopped\n"
        node.close()
        assert not list(tmp_path.iterdir())
        assert disk.used == arena.budget.used == (0, 0)
    finally:
        node.close()
        arena.close()
    assert admission.used.claims == 0


def test_actual_workers_borrow_their_full_host_demand_from_one_parent_claim(
    tmp_path, monkeypatch
):
    cpu_workers(monkeypatch)
    node, admission, transfers, disk = configured(tmp_path)
    host_bytes = node._request.host_bytes
    with HostArena(admission, host_bytes, 1) as arena:
        host = arena.partition(host_bytes, 1)
        node._host_memory = host
        try:
            processes = node.start(lambda _: io.BytesIO(PAYLOAD), startup_timeout=5)
            with start_processes(processes, stop=node.stop):
                assert node.lease.request.host_bytes == host_bytes
                assert admission.used.host_bytes == (64 << 20) + host_bytes
                assert host.budget.used == (host_bytes, 1)
            assert host.budget.used == (0, 0)
            assert admission.used.host_bytes == (64 << 20) + host_bytes
            node.close()
        finally:
            node.close()
            host.close()
            transfers.close()
    assert admission.used.claims == 0 and disk.used == (0, 0)


@pytest.mark.parametrize("fault", ["disk", "checksum", "second_launch"])
def test_partial_start_retains_all_owners_until_explicit_cleanup(
    tmp_path, monkeypatch, fault
):
    cpu_workers(monkeypatch)
    node, admission, arena, disk = configured(
        tmp_path, disk_bytes=100 if fault == "disk" else 16 << 20
    )
    original_launch = node_module.launch_worker
    launches = []

    def launch(*args, **kwargs):
        rank = args[2].startup.scope.rank
        launches.append(rank)
        if rank == 1:
            raise OSError("second launch failed")
        return original_launch(*args, **kwargs)

    monkeypatch.setattr(node_module, "launch_worker", launch)
    try:
        with pytest.raises((ValueError, ProtocolError, OSError)):
            node.start(
                lambda _: io.BytesIO(b"wrong" if fault == "checksum" else PAYLOAD)
            )
        assert not node.launched and not node.closed
        assert admission.used.claims == 2
        assert launches == ([0, 1] if fault == "second_launch" else [])
        node.close()
        assert node.lease.closed and disk.used == arena.budget.used == (0, 0)
        assert not list(tmp_path.iterdir())
    finally:
        node.close()
        arena.close()


def test_active_result_owner_blocks_file_cleanup_after_workers_stop(
    tmp_path, monkeypatch
):
    cpu_workers(monkeypatch)
    node, admission, arena, disk = configured(tmp_path)
    try:
        processes = node.start(lambda _: io.BytesIO(PAYLOAD), startup_timeout=5)
        group = start_processes(processes, stop=node.stop)
        output = node.files.outputs.reserve(OutputSpec("result", 10, "bytes"))
        group.close()
        with pytest.raises(ExceptionGroup, match="files still require cleanup"):
            node.close()
        assert not node.closed and node.lease.closed
        assert node._directory is not None
        assert output._owner.root.exists()
        assert disk.used[0] > 0 and arena.budget.used[0] > 0
        output.close()
        node.close()
        assert disk.used == arena.budget.used == (0, 0)
    finally:
        node.close()
        arena.close()
    assert admission.used.claims == 0


def test_ungranted_request_delivery_outlives_private_node_directory(
    tmp_path, monkeypatch
):
    cpu_workers(monkeypatch)
    node, admission, arena, disk = configured(tmp_path)
    try:
        processes = node.start(lambda _: io.BytesIO(PAYLOAD), startup_timeout=5)
        with start_processes(processes, stop=node.stop):
            delivery = FileDelivery(
                [REFERENCE], root=node.files.root, disk=disk, memory=arena.budget
            )
            delivery.receive(lambda _: io.BytesIO(PAYLOAD))
        node.close()
        # A request can fail between its copy and its grant. Its separate owner,
        # not node shutdown, controls that delivery's root and reservations.
        with delivery.use() as resolve:
            assert resolve(REFERENCE).read_bytes() == PAYLOAD
        with pytest.raises(RuntimeError, match="live buffer owners"):
            arena.close()
        delivery.close()
        assert disk.used == arena.budget.used == (0, 0)
    finally:
        node.close()
        arena.close()
    assert admission.used.claims == 0
