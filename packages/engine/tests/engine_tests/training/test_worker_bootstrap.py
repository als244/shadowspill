"""Installed worker startup stays CPU-only until exact all-worker release."""

import hashlib
import io
import os
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from engine_tests.training.test_worker_loop import call
from engine_tests.training.test_worker_startup import BUNDLE, SCOPE, prepared_worker
from shadowspill.orchestration.launch.admission import (
    AssignedDevice,
    NodeAdmission,
    NodeCapacity,
    ResourceRequest,
)
from shadowspill.orchestration.launch.file_grants import FileGrants, verify_file
from shadowspill.orchestration.launch.log_files import LogFiles
from shadowspill.orchestration.launch.staging import FileDelivery
from shadowspill.training.worker import bootstrap
from shadowspill.training.worker.channel import write_message
from shadowspill.training.worker.launch import launch_worker, start_processes
from shadowspill.training.worker.protocol import CODEC, WorkerLaunch
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.codec import ProtocolError
from shadowspill_client.protocol.requests import ArtifactRef


def test_launch_record_is_bounded_frozen_and_exclusive_to_the_private_codec():
    startup = prepared_worker().startup
    weights = [ArtifactRef("weights", "b" * 64, 10)]
    launch = WorkerLaunch(startup, ["fixture"], weights)
    assert CODEC.decode(CODEC.encode(launch)) == launch
    weights.clear()
    assert len(launch.weights) == 1 and launch.packages == ("fixture",)
    with pytest.raises(ValueError, match="distinct"):
        WorkerLaunch(startup, (), [startup.bundle])
    for packages in (["name"] * 2, ["x" * 129], [str(i) for i in range(65)]):
        with pytest.raises(ValueError):
            WorkerLaunch(startup, packages)


@pytest.mark.parametrize("issue", ["initial", "preflight", "weights", None])
def test_bootstrap_validates_grants_and_preflight_before_opening_executor(
    tmp_path, monkeypatch, issue
):
    expected = prepared_worker().startup
    launch = WorkerLaunch(expected, ("fixture",))
    incoming = io.BytesIO()
    write_message(incoming, call() if issue == "initial" else launch)
    incoming.seek(0)
    events = []

    def load(packages):
        assert packages == ("fixture",)
        events.append("definitions")
        return "registry"

    def verify(root, reference):
        assert root == tmp_path and reference == expected.bundle
        events.append("integrity")
        return tmp_path / "bundle"

    def recheck(registry):
        assert registry == "registry"
        events.append("bundle")
        return "family"

    def serve(reader, writer, runtime, resolve, outputs, *, traces, capture_progress):
        assert reader is writer is incoming
        assert capture_progress is None
        assert runtime.cache_root == outputs.root == tmp_path
        assert traces.root == tmp_path and traces.bundle == expected.bundle
        with pytest.raises(ValueError, match="ungranted"):
            runtime.artifact("unknown")
        events.append("serve")

    def prepared(family, scope, **options):
        assert family == "family" and scope == SCOPE
        assert options == {"bundle": BUNDLE, "rank_map": {0: 0}, "restore": None}
        events.append("preflight")
        return SimpleNamespace(
            startup=replace(expected, staging_bytes=1)
            if issue == "preflight"
            else expected,
            initialization=SimpleNamespace(
                context=SimpleNamespace(
                    weights=SimpleNamespace(kind="artifact", reference="missing")
                    if issue == "weights"
                    else SimpleNamespace(kind="random")
                )
            ),
            outputs={},
            traces={},
            serve=serve,
        )

    monkeypatch.setattr(bootstrap, "load_registry", load)
    monkeypatch.setattr(bootstrap, "verify_file", verify)
    monkeypatch.setattr(
        bootstrap.TrainingBundle, "load", lambda _: SimpleNamespace(recheck=recheck)
    )
    monkeypatch.setattr(bootstrap.PreparedWorker, "resolve", prepared)
    if issue:
        with pytest.raises((ValueError, ProtocolError)):
            bootstrap.serve(incoming, files=tmp_path, outputs=tmp_path, cache=tmp_path)
        assert "serve" not in events
        if issue == "initial":
            assert not events
    else:
        bootstrap.serve(incoming, files=tmp_path, outputs=tmp_path, cache=tmp_path)
        assert events == ["definitions", "integrity", "bundle", "preflight", "serve"]


def test_startup_artifact_content_is_rechecked_before_decoding(tmp_path):
    payload = b"x" * (128 << 10)
    reference = ArtifactRef("bundle", hashlib.sha256(payload).hexdigest(), len(payload))
    disk, memory = BufferBudget(1 << 20, 100), BufferBudget(4 << 20, 100)
    with (
        FileDelivery([reference], root=tmp_path, disk=disk, memory=memory) as delivery,
        FileGrants(root=tmp_path, memory=memory) as grants,
    ):
        delivery.receive(lambda _: io.BytesIO(payload))
        with grants.lend(delivery, [reference]):
            path = verify_file(grants.root, reference)
            path.write_bytes(b"z" * len(payload))
            with pytest.raises(ValueError, match="checksum"):
                verify_file(grants.root, reference)
    assert disk.used == memory.used == (0, 0)


def test_real_owned_bootstrap_failure_releases_no_claim_until_child_cleanup(tmp_path):
    expected = prepared_worker().startup
    admission = NodeAdmission(
        NodeCapacity("local", (AssignedDevice("gpu", "0", 2 << 30),), 4 << 30)
    )
    lease = admission.reserve(ResourceRequest({"gpu": 2 << 30}, 1 << 30))
    disk, memory = BufferBudget(1 << 20, 100), BufferBudget(1 << 20, 100)
    logs = LogFiles(
        lease, "gpu", root=tmp_path, disk=disk, memory=memory, limit_bytes=64 << 10
    )
    try:
        process = launch_worker(
            lease,
            "gpu",
            WorkerLaunch(expected, ("shadowspill-definitely-not-installed",)),
            environment=os.environ,
            files=tmp_path,
            outputs=tmp_path,
            cache=tmp_path,
            logs=logs,
            cwd=tmp_path,
            startup_timeout=5,
        )
        assert admission.used.claims == 1
        with pytest.raises(ProtocolError, match="startup evidence"):
            start_processes({0: process}, stop=lease.close)
        assert process.child.closed and lease.closed and admission.used.claims == 0
        assert (
            "definition package is not installed"
            in (logs.directory / "stderr.log").read_text()
        )
        assert not process.child._ready
    finally:
        lease.close()
        logs.close()
    assert disk.used == memory.used == (0, 0)


@pytest.mark.parametrize("issue", ["device", "host", "rendezvous"])
def test_launch_rejects_unadmitted_pools_and_missing_rendezvous_before_spawn(
    tmp_path, issue
):
    expected = prepared_worker().startup
    if issue == "rendezvous":
        expected = replace(expected, logical_by_worker=(0, 1))
    request = ResourceRequest(
        {"gpu": 1 if issue == "device" else 2 << 30}, 1 if issue == "host" else 1 << 30
    )
    admission = NodeAdmission(
        NodeCapacity("local", (AssignedDevice("gpu", "0", 2 << 30),), 4 << 30)
    )
    lease = admission.reserve(request)
    logs = LogFiles(
        lease,
        "gpu",
        root=tmp_path,
        disk=BufferBudget(1 << 20, 100),
        memory=BufferBudget(1 << 20, 100),
        limit_bytes=64 << 10,
    )
    with logs, lease, pytest.raises(ValueError):
        launch_worker(
            lease,
            "gpu",
            WorkerLaunch(expected, ()),
            environment={},
            files=tmp_path,
            outputs=tmp_path,
            cache=tmp_path,
            logs=logs,
            cwd=tmp_path,
        )


def test_agent_launch_imports_without_framework_provider_or_workloads():
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    blocked = {'torch', 'triton', 'pynvml', 'shadowspill_workloads'}
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in self.blocked:
            raise AssertionError('unexpected agent import: ' + fullname)
sys.meta_path.insert(0, Block())
from shadowspill.training.worker.launch import launch_worker, start_processes
from shadowspill.training.worker.bootstrap import serve
from shadowspill.training.worker.node import WorkerNode
from shadowspill.training.session.controller import RunController
from shadowspill.training.inputs.frame_reader import FrameReader
from shadowspill.training.result_tensors import write_tensor_region
from shadowspill.server.results import ResultPublisher
""",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
