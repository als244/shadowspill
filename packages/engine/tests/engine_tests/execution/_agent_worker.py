"""Framework-free controller, actual installed GPU child and owned binary files."""

import hashlib
import importlib.abc
import io
import json
import os
import sys
from pathlib import Path


class Block(importlib.abc.MetaPathFinder):
    blocked = frozenset({"torch", "triton", "pynvml", "shadowspill_workloads"})

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.blocked:
            raise AssertionError("GPU/catalog import in controller: " + fullname)


sys.meta_path.insert(0, Block())

import numpy as np  # noqa: E402

from engine_tests.fixtures.definition_operations import records  # noqa: E402
from shadowspill.orchestration.launch.admission import (  # noqa: E402
    AssignedDevice,
    NodeAdmission,
    NodeCapacity,
    ResourceRequest,
)
from shadowspill.orchestration.launch.device_inventory import probe_devices  # noqa: E402
from shadowspill.orchestration.launch.file_grants import FileGrants  # noqa: E402
from shadowspill.orchestration.launch.host_memory import host_memory  # noqa: E402
from shadowspill.orchestration.launch.publication import (  # noqa: E402
    OutputFiles,
    require_output,
)
from shadowspill.orchestration.launch.staging import FileDelivery  # noqa: E402
from shadowspill.server.records.requests import RequestRecords, RunIdentity  # noqa: E402
from shadowspill.training.definitions import load_registry  # noqa: E402
from shadowspill.training.preparation.bundle import TrainingBundle  # noqa: E402
from shadowspill.training.preparation.job import JOB_NAME, PreparationJob  # noqa: E402
from shadowspill.training.session.dispatch import RequestDispatcher  # noqa: E402
from shadowspill.training.session.inputs import RequestInputs  # noqa: E402
from shadowspill.training.session.prepared import PreparedRequests  # noqa: E402
from shadowspill.training.session.state import OperationState  # noqa: E402
from shadowspill.training.worker.launch import (  # noqa: E402
    launch_worker,
    start_processes,
)
from shadowspill.training.worker.protocol import WorkerLaunch, WorkerScope  # noqa: E402
from shadowspill.training.worker.requests import NodeFiles, WorkerRequests  # noqa: E402
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
expected = prepared_worker.startup
devices = probe_devices(environment=os.environ, root=directory)
assert len(devices.devices) == 1
device = devices.devices[0]
admission = NodeAdmission(
    NodeCapacity(
        "local",
        (AssignedDevice(device.uuid, device.uuid, device.total_bytes),),
        host_memory().available_bytes,
    )
)
# This tiny integration reserves a conservative explicit test envelope. General
# process/decoded-artifact overhead policy remains a separate service gate.
lease = admission.reserve(
    ResourceRequest({device.uuid: expected.device_bytes}, 8 << 30)
)
disk, memory = BufferBudget(128 << 20, 100), BufferBudget(64 << 20, 100)
metadata = BufferBudget(16 << 20)
oracle = json.loads((root / "agent-oracle.json").read_text())

with (
    FileDelivery([reference], root=directory, disk=disk, memory=memory) as bundle_files,
    FileGrants(root=directory, memory=memory) as grants,
    OutputFiles(root=directory, disk=disk, memory=memory) as outputs,
):
    bundle_files.receive(lambda _: (root / "training.ssp").open("rb"))
    with grants.lend(bundle_files, [reference]), lease:
        process = launch_worker(
            lease,
            device.uuid,
            WorkerLaunch(expected, packages),
            environment={
                **os.environ,
                "TORCHINDUCTOR_CACHE_DIR": str(directory / "inductor"),
                "TRITON_CACHE_DIR": str(directory / "triton"),
            },
            files=grants.root,
            outputs=outputs.root,
            cache=directory,
            directory=directory / "worker",
            cwd=directory,
            log_limit_bytes=64 << 10,
        )
        with (
            start_processes({0: process}, stop=lease.close) as group,
            WorkerRequests(
                group,
                {0: prepared_worker},
                {0: NodeFiles(directory, disk, memory, grants, outputs)},
                metadata=metadata,
            ) as transfer,
            RequestRecords(
                directory / "requests",
                identity=RunIdentity("owner", "session", "run"),
                cache_root=directory,
            ) as journal,
        ):
            assert process.child._ready and process.child._socket.gettimeout() is None
            state = OperationState()
            dispatcher = RequestDispatcher(journal, state, max_pending=3)
            prepared = PreparedRequests(family, cache_root=directory)
            journal.claim_writer("writer", "agent")
            inputs = RequestInputs(
                journal, dispatcher.order, prepared.setup.config.execution, prepared
            )

            def invoke(operation, *, predictions=None, corrupt=False, **options):
                sequence = journal.last_sequence + 1
                name = f"request-{sequence}"
                payload, wire, count = None, None, 0
                if operation != "optim_step":
                    datums = records(targets=operation != "forward")
                    with snapshot_batch(datums, BufferBudget(1 << 20)) as snapshot:
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
                prepared.register(inputs, request)
                status = inputs.receive(
                    name, None if wire is None else io.BytesIO(wire)
                )
                assert status.state == "accepted", status
                data = inputs.take(name)

                def execute(transition):
                    return transfer.execute(data, transition)

                def publish(receipt, result):
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
                            BufferBudget(frame.frame.size.staging),
                            limits=frame.frame.size,
                        ) as decoded,
                    ):
                        rows = decoded.datums[0].inputs
                        if operation == "step":
                            loss = sum(
                                row["loss"].array().item() for row in rows.values()
                            )
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
                                actual[piece.destination.datum][
                                    piece.destination_axes
                                ] = buffers[piece.buffer][piece.source_axes]
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

                try:
                    status = dispatcher.run(
                        name, execute, publish, participates=data.participates
                    )
                    if corrupt:
                        assert status.failure.code == "publication_failed", status
                        assert status.failure.execution_effect == "completed"
                        assert status.failure.state_valid
                        assert status.failure.completed_updates == 2
                        assert (
                            dispatcher.run(name, lambda _: pytest_fail(), publish)
                            == status
                        )
                    else:
                        assert status.state == "completed", status
                finally:
                    inputs.finish(name)
                assert inputs.staging.used == (0, 0)

            def pytest_fail():
                raise AssertionError("completed optimizer replayed")

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
            assert state.boundary.completed_updates == 2 and state.boundary.valid
        assert process.child.closed and lease.closed and admission.used.claims == 0
assert disk.used == memory.used == metadata.used == (0, 0)
assert not any(name in sys.modules for name in ("torch", "triton", "pynvml"))
print("Qualified installed agent-owned GPU bootstrap and binary publication")
print("Qualified production worker request/file coordination")
