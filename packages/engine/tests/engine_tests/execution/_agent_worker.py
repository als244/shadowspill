"""Framework-free controller, actual installed GPU child and owned binary files."""

import hashlib
import importlib.abc
import io
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path


class Block(importlib.abc.MetaPathFinder):
    blocked = frozenset({"torch", "triton", "pynvml", "shadowspill_workloads"})

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.blocked:
            raise AssertionError("GPU/catalog import in controller: " + fullname)


sys.meta_path.insert(0, Block())

import numpy as np  # noqa: E402

from engine_tests.fixtures.definition_operations import records  # noqa: E402
from shadowspill.orchestration.launch.admission import NodeAdmission  # noqa: E402
from shadowspill.orchestration.launch.host_arena import HostArena  # noqa: E402
from shadowspill.orchestration.launch.node_capacity import discover_capacity  # noqa: E402
from shadowspill.orchestration.launch.publication import require_output  # noqa: E402
from shadowspill.orchestration.launch.staging import FileDelivery  # noqa: E402
from shadowspill.server.records.requests import RequestRecords, RunIdentity  # noqa: E402
from shadowspill.training.definitions import load_registry  # noqa: E402
from shadowspill.training.preparation.bundle import TrainingBundle  # noqa: E402
from shadowspill.training.preparation.job import JOB_NAME, PreparationJob  # noqa: E402
from shadowspill.training.session.controller import RunController  # noqa: E402
from shadowspill.training.session.prepared import PreparedRequests  # noqa: E402
from shadowspill.training.worker.launch import start_processes  # noqa: E402
from shadowspill.training.worker.node import WorkerNode  # noqa: E402
from shadowspill.training.worker.protocol import WorkerScope  # noqa: E402
from shadowspill.training.worker.requests import WorkerRequests  # noqa: E402
from shadowspill.training.worker.resources import WorkerHost  # noqa: E402
from shadowspill.training.worker.runtime import PreparedWorker  # noqa: E402
from shadowspill_client.buffers import BufferBudget  # noqa: E402
from shadowspill_client.protocol.payload import MEDIA_TYPE, snapshot_batch  # noqa: E402
from shadowspill_client.protocol.payload_io import read_payload  # noqa: E402
from shadowspill_client.protocol.requests import (  # noqa: E402
    ArtifactRef,
    ComputeRequest,
    RequestIdentity,
)
from shadowspill_client.protocol.results import ComputeResult  # noqa: E402

root = Path(sys.argv[1])
directory = root / "agent"
directory.mkdir()
packages = PreparationJob.load(root / JOB_NAME).packages
family = TrainingBundle.load(root / "training.ssp").recheck(load_registry(packages))
with (root / "training.ssp").open("rb") as stream:
    reference = ArtifactRef(
        "bundle",
        hashlib.file_digest(stream, "sha256").hexdigest(),
        (root / "training.ssp").stat().st_size,
    )
prepared_worker = PreparedWorker.resolve(
    family, WorkerScope("run", "agent", "gpu-1", 0), bundle=reference
)
controller_allowance = 1 << 30
admission = NodeAdmission(
    discover_capacity(
        environment=os.environ,
        root=directory,
        node_id="local",
        reserved_host_bytes=controller_allowance,
    )
)
assert len(admission.capacity.devices) == 1
device = admission.capacity.devices[0]
# These explicit deployment allowances are qualified for this installed tiny
# fixture only. Record kernel high-water observations without calling them a
# maximum proof for arbitrary definitions or larger bundles.
host_allowance = WorkerHost(2 << 30, 512 << 20)
disk = BufferBudget(128 << 20, 100)
execution = prepared_worker.bindings.setup.config.execution
oracle = json.loads((root / "agent-oracle.json").read_text())
observations = []


def host_peak(pid):
    with Path(f"/proc/{pid}/status").open("rb") as stream:
        raw = stream.read((64 << 10) + 1)
    assert len(raw) <= 64 << 10
    for line in raw.decode().splitlines():
        if line.startswith("VmHWM:"):
            _, amount, unit = line.split()
            assert unit == "kB"
            return int(amount) * 1024
    raise AssertionError("owned process has no resident high-water observation")


def observe(phase, process):
    worker = host_peak(process.child.process.pid)
    controller = host_peak(os.getpid())
    assert worker <= lease.request.host_bytes
    assert controller <= controller_allowance + arena_bytes
    observations.append(
        {
            "phase": phase,
            "worker_peak_bytes": worker,
            "controller_peak_bytes": controller,
        }
    )


@contextmanager
def owned_node(memory):
    node = WorkerNode(
        admission,
        {0: prepared_worker},
        {0: device.id},
        {0: host_allowance},
        packages=packages,
        root=directory,
        cache=directory,
        disk=disk,
        memory=memory,
        environment={
            **os.environ,
            "TORCHINDUCTOR_CACHE_DIR": str(directory / "inductor"),
            "TRITON_CACHE_DIR": str(directory / "triton"),
        },
        log_limit_bytes=64 << 10,
    )
    try:
        yield node
    finally:
        node.stop()
        if sys.exc_info()[0] is not None:
            for logs in node.logs.values():
                path = logs.directory / "stderr.log"
                if path.exists():
                    with path.open("rb") as stream:
                        sys.stderr.write(stream.read(64 << 10).decode(errors="replace"))
        node.close()


with (
    HostArena(admission, 64 << 20, 100) as transfers,
    HostArena(admission, 16 << 20, 2) as commands,
    HostArena(admission, 64 << 20, 2) as publication,
    HostArena(
        admission, execution.max_staging_bytes, execution.max_pending_requests
    ) as uploads,
    owned_node(transfers.budget) as node,
):
    memory, metadata = transfers.budget, commands.budget
    arena_bytes = (144 << 20) + execution.max_staging_bytes
    assert admission.used.host_bytes == arena_bytes
    processes = node.start(lambda _: (root / "training.ssp").open("rb"))
    process, lease = processes[0], node.lease
    outputs, logs = node.files.outputs, node.logs[0]
    assert admission.used.host_bytes == lease.request.host_bytes + arena_bytes
    prepared = PreparedRequests(family, cache_root=directory)
    publishers = {}
    with (
        start_processes(processes, stop=node.stop) as group,
        WorkerRequests(
            group,
            {0: prepared_worker},
            {0: node.files},
            metadata=metadata,
        ) as transfer,
        RequestRecords(
            directory / "requests",
            identity=RunIdentity("owner", "session", "run"),
            cache_root=directory,
        ) as journal,
        RunController(
            journal,
            prepared,
            transfer,
            lambda call, receipt, result: publishers[call.request.identity.request_id](
                call, receipt, result
            ),
            server_incarnation="agent",
            budget=uploads.budget,
        ) as controller,
    ):
        assert process.child._ready and process.child._socket.gettimeout() is None
        observe("ready", process)
        assert controller.attach("writer").ready

        def invoke(operation, *, predictions=None, corrupt=False, **options):
            sequence = journal.last_sequence + 1
            name = f"request-{sequence}"
            payload, wire, count = None, None, 0
            if operation != "optim_step":
                datums = records(targets=operation != "forward")
                with snapshot_batch(datums, publication.budget) as snapshot:
                    payload = ArtifactRef(
                        name, snapshot.sha256, snapshot.size_bytes, MEDIA_TYPE
                    )
                    wire, count = b"".join(snapshot.chunks()), len(datums)
            request = ComputeRequest(
                RequestIdentity(name, "run", sequence, "writer", 1, "agent"),
                operation,
                payload,
                count,
                **options,
            )
            def publish(data, receipt, result):
                assert journal.execution(name) == receipt
                if corrupt:
                    path = require_output(outputs.root, result._files[0].spec)
                    with path.open("r+b", buffering=0) as stream:
                        stream.write(b"corrupt!")
                result.verify()
                frame = result.frames[0]
                with (
                    result.open(0) as stream,
                    read_payload(
                        stream,
                        result.replies[0].result,
                        publication.budget,
                        limits=frame.frame.size,
                    ) as decoded,
                ):
                    rows = decoded.datums[0].inputs
                    if operation == "step":
                        loss = sum(row["loss"].array().item() for row in rows.values())
                        np.testing.assert_allclose(loss, oracle["loss"], rtol=1e-4)
                    if predictions:
                        actual = np.empty((3, 3), dtype=np.float32)
                        buffers = {
                            buffer: rows[str(index)]["values"]["prediction"].array()
                            for index, buffer in enumerate(frame.buffers)
                        }
                        for piece in data.candidate.source.layout.outputs.pieces(
                            data.assignment
                        ):
                            actual[piece.destination.datum][piece.destination_axes] = (
                                buffers[piece.buffer][piece.source_axes]
                            )
                        np.testing.assert_allclose(
                            actual, oracle[predictions], rtol=1e-4, atol=1e-5
                        )
                return ComputeResult(
                    request.identity,
                    operation,
                    "history",
                    receipt.outcome,
                    receipt.completed_updates,
                    receipt.accumulation,
                    optim=data.controls.optim,
                    loss_params=data.controls.loss_params,
                    grad_scale=data.controls.grad_scale,
                )

            publishers[name] = publish
            controller.register(request)
            with io.BytesIO(wire or b"") as body:
                status = controller.receive(name, None if wire is None else body)
            assert status.state == "accepted", status
            # The submitting body is already closed; only the server queue owns
            # execution. Waiting reads records and cannot issue another command.
            status = controller.wait(name, timeout=120)
            controller.drain(timeout=120)
            del publishers[name]
            if corrupt:
                assert status.failure.code == "publication_failed", status
                assert status.failure.execution_effect == "completed"
                assert status.failure.state_valid
                assert status.failure.completed_updates == 2
                assert (
                    controller.register(request) == controller.receive(name) == status
                )
            else:
                assert status.state == "completed", status
            assert controller.inputs.staging.used == (0, 0)
            observe(name, process)

        invoke(
            "step",
            optim={"lr": 0.03},
            grad_scale=0.5,
            loss_params={"temperature": 2.0},
        )
        invoke("forward", outputs="predictions", predictions="after_step")
        invoke("forward_backward", loss_params={"temperature": 0.5})
        invoke("forward_backward", loss_params={"temperature": 1.0})
        invoke("optim_step", optim={"lr": 0.02}, grad_scale=-0.5, corrupt=True)
        invoke("forward", outputs="predictions", predictions="after_split")
        assert controller.boundary.completed_updates == 2 and controller.boundary.valid
    assert process.child.closed and lease.closed
    assert admission.used.host_bytes == arena_bytes
    log_refs = logs.references()
    with FileDelivery(
        list(log_refs.values()), root=directory, disk=disk, memory=memory
    ) as saved_logs:
        saved_logs.receive(logs.open)
        logs.close()
        with saved_logs.use() as resolve:
            for reference in log_refs.values():
                assert resolve(reference).stat().st_size == reference.size_bytes
assert admission.used.claims == 0
assert disk.used == memory.used == metadata.used == publication.budget.used == (0, 0)
(directory / "host-accounting.json").write_text(
    json.dumps(
        {
            "worker_host_claim_bytes": lease.request.host_bytes,
            "controller_fixed_allowance_bytes": controller_allowance,
            "agent_arena_claim_bytes": arena_bytes,
            "observations": observations,
            "all_claims_released": admission.used.claims == 0,
        },
        indent=2,
    )
    + "\n"
)
assert not any(name in sys.modules for name in ("torch", "triton", "pynvml"))
print("Qualified installed agent-owned GPU bootstrap and binary publication")
print("Qualified production worker request/file coordination")
print("Qualified node-backed worker and input/artifact host accounting")
print("Qualified pre-reserved worker logs and post-exit artifact transfer")
print("Qualified production node worker lifecycle")
print("Qualified connection-independent production controller queue")
