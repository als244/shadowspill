"""Framework-free controller, actual installed GPU child and owned binary files."""

import hashlib
import importlib.abc
import io
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from threading import Thread


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
from shadowspill.server.authentication import BearerTokens  # noqa: E402
from shadowspill.server.http import HTTP_SCRATCH_BYTES, HTTPService, RunAccess  # noqa: E402
from shadowspill.server.records.requests import RunIdentity  # noqa: E402
from shadowspill.server.results import ResultPublisher  # noqa: E402
from shadowspill.server.storage import ServiceStorage  # noqa: E402
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
from shadowspill_client.protocol.codec import encode  # noqa: E402
from shadowspill_client.protocol.payload import MEDIA_TYPE, snapshot_batch  # noqa: E402
from shadowspill_client.protocol.payload_io import read_payload  # noqa: E402
from shadowspill_client.protocol.requests import (  # noqa: E402
    ArtifactRef,
    OperationRequest,
    RequestIdentity,
)
from shadowspill_client.protocol.service import ServiceError  # noqa: E402
from shadowspill_client.transport.http import HTTPTransport  # noqa: E402

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
bundle_reference = reference
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
published = []
token = "integration-" + os.urandom(32).hex()
route = "/v1/runs/session/run"
run_identity = RunIdentity("owner", "session", "run")


def persistent_storage(memory, *, incarnation="agent"):
    storage = ServiceStorage(
        directory / "service",
        cache_root=directory,
        incarnation=incarnation,
        memory=memory,
        request_host_bytes=128 << 20,
        max_request_owners=1,
        max_artifact_owners=1,
        artifact_bytes=128 << 20,
        artifact_count=100,
    )
    storage.recover()
    return storage


@contextmanager
def new_storage(memory):
    with persistent_storage(memory) as storage, storage.artifacts("owner") as artifacts:
        config = encode(prepared_worker.bindings.setup.config)
        config_ref = ArtifactRef(
            "configuration", hashlib.sha256(config).hexdigest(), len(config)
        )
        artifacts.put(config_ref, lambda _: io.BytesIO(config))
        artifacts.put(bundle_reference, lambda _: (root / "training.ssp").open("rb"))
        _, created = storage.catalog.create(
            run_identity, request_sha256=config_ref.sha256, configuration=config_ref
        )
        assert created
        storage.catalog.progress(run_identity, "initializing")
        yield storage, artifacts


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
def owned_node(memory, worker=prepared_worker):
    node = WorkerNode(
        admission,
        {0: worker},
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


@contextmanager
def serving(controller, storage, memory, client_memory):
    @contextmanager
    def runs(identity):
        assert identity == run_identity
        with storage.requests(identity) as journal:
            yield RunAccess(journal, controller)

    service = HTTPService(
        ("127.0.0.1", 0),
        incarnation="agent",
        authentication=BearerTokens({token: "owner"}),
        runs=runs,
        artifacts=storage.artifacts,
        memory=memory,
    )
    thread = Thread(target=service.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    client = HTTPTransport(
        f"http://127.0.0.1:{service.server_port}",
        token=token,
        memory=client_memory,
    )
    try:
        yield client
    finally:
        service.shutdown()
        service.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert memory.used == client_memory.used == (0, 0)


with (
    HostArena(admission, 64 << 20, 100) as transfers,
    HostArena(admission, 16 << 20, 2) as commands,
    # One assembly owner, one frame reader and one reduction scratch owner.
    HostArena(admission, 64 << 20, 3) as publication,
    HostArena(admission, HTTP_SCRATCH_BYTES * 2, 2) as http_server,
    HostArena(admission, HTTP_SCRATCH_BYTES, 1) as http_client,
    HostArena(admission, 160 << 20, 8) as storage_arena,
    HostArena(
        admission, execution.max_staging_bytes, execution.max_pending_requests
    ) as uploads,
    new_storage(storage_arena.budget) as (storage, artifacts),
    owned_node(transfers.budget) as node,
):
    memory, metadata = transfers.budget, commands.budget
    arena_bytes = (304 << 20) + execution.max_staging_bytes + 3 * HTTP_SCRATCH_BYTES
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
        storage.requests(run_identity) as journal,
        ResultPublisher(
            artifacts,
            session_id="session",
            history_id="history",
            root=directory,
            memory=publication.budget,
            disk=disk,
        ) as result_publisher,
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
        serving(controller, storage, http_server.budget, http_client.budget) as client,
    ):
        assert process.child._ready and process.child._socket.gettimeout() is None
        storage.catalog.ready(
            run_identity, plans=bundle_reference, initial_history_id="history"
        )
        observe("ready", process)
        assert client.request("GET", "/v1/info").value.owner_id == "owner"
        assert client.request("POST", route + "/writers/writer").status == 200

        def invoke(operation, *, predictions=None, corrupt=False, **options):
            sequence = journal.last_sequence + 1
            name = f"request-{sequence}"
            payload, wire, count = None, None, 0
            if operation not in ("optim_step", "discard", "close"):
                datums = records(targets=operation != "forward")
                with snapshot_batch(datums, publication.budget) as snapshot:
                    payload = ArtifactRef(
                        name, snapshot.sha256, snapshot.size_bytes, MEDIA_TYPE
                    )
                    wire, count = b"".join(snapshot.chunks()), len(datums)
            request = OperationRequest(
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
                outcome = result_publisher(data, receipt, result)
                if operation == "step":
                    np.testing.assert_allclose(outcome.loss, oracle["loss"], rtol=1e-4)
                if predictions:
                    ref = outcome.outputs["prediction"]
                    with (
                        artifacts.open(ref) as stream,
                        read_payload(stream, ref, publication.budget) as decoded,
                    ):
                        rows = decoded.datums[0].inputs
                        actual = np.stack(
                            [rows[str(index)].array() for index in range(3)]
                        )
                        np.testing.assert_allclose(
                            actual, oracle[predictions], rtol=1e-4, atol=1e-5
                        )
                    published.append((ref, None))
                return outcome

            publishers[name] = publish
            client.request("POST", route + "/requests", request)
            input_path = route + "/requests/" + name + "/input"
            status = (
                client.request("PUT", input_path)
                if wire is None
                else client.upload(input_path, payload, [wire])
            ).value
            assert status.acceptance is not None, status
            # Each HTTP connection is already closed; only the server queue owns
            # execution. Polling reads records and cannot issue another command.
            for _ in range(12):
                status = client.request(
                    "GET", route + "/requests/" + name + "?wait=completed&timeout=10"
                ).value
                if status.state in ("completed", "skipped", "failed"):
                    break
            else:
                raise AssertionError("worker request exceeded the integration deadline")
            controller.drain(timeout=120)
            del publishers[name]
            if corrupt:
                assert status.failure.code == "publication_failed", status
                assert status.failure.execution_effect == "completed"
                assert status.failure.state_valid
                assert status.failure.completed_updates == 2
                assert (
                    client.request("POST", route + "/requests", request).value == status
                )
            else:
                assert status.state == "completed", status
            assert controller.inputs.staging.used == (0, 0)
            if operation != "close":
                observe(name, process)
            else:
                assert process.child.closed and lease.closed and controller.closed
                assert (
                    client.request("POST", route + "/requests", request).value == status
                )

        invoke(
            "step",
            optim={"lr": 0.03},
            grad_scale=0.5,
            loss_params={"temperature": 2.0},
        )
        invoke("forward", outputs="predictions", predictions="after_step")
        invoke("forward_backward", loss_params={"temperature": 2.0})
        invoke("discard")
        assert not controller.boundary.accumulation.open
        assert controller.boundary.completed_updates == 1
        invoke("forward", outputs="predictions", predictions="after_step")
        invoke("forward_backward", loss_params={"temperature": 0.5})
        invoke("forward_backward", loss_params={"temperature": 1.0})
        invoke("optim_step", optim={"lr": 0.02}, grad_scale=-0.5, corrupt=True)
        invoke("forward", outputs="predictions", predictions="after_split")
        assert controller.boundary.completed_updates == 2 and controller.boundary.valid
        invoke("close")
        assert controller.boundary.closed and controller.closed
        storage.catalog.finish(run_identity)
        # Completed handles remain downloadable after execution workers exit.
        for reference, _ in published:
            downloaded = io.BytesIO()
            client.download("/v1/artifacts/" + reference.id, reference, downloaded)
            with read_payload(
                io.BytesIO(downloaded.getvalue()), reference, publication.budget
            ) as decoded:
                assert decoded.datums
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
    node.close()
    assert published and not outputs.root.exists()
    for reference, limits in published:
        with (
            artifacts.open(reference) as stream,
            read_payload(
                stream, reference, publication.budget, limits=limits
            ) as decoded,
        ):
            assert decoded.datums  # Valid binary payload after worker-file deletion.
assert admission.used.claims == 0
assert disk.used == memory.used == metadata.used == publication.budget.used == (0, 0)
assert storage_arena.budget.used == (0, 0)
with (
    HostArena(admission, 160 << 20, 8) as reopened_arena,
    persistent_storage(reopened_arena.budget, incarnation="replacement") as reopened,
    reopened.requests(run_identity) as saved_requests,
    reopened.artifacts("owner") as saved_artifacts,
):
    assert reopened.catalog.get(run_identity).state == "closed"
    assert (
        saved_requests.get(f"request-{saved_requests.last_sequence}")[
            1
        ].result.operation
        == "close"
    )
    for reference, _ in published:
        with saved_artifacts.open(reference) as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == reference.sha256
assert admission.used.claims == 0 and reopened_arena.budget.used == (0, 0)


def qualify_allocation_stop():
    identity = RunIdentity("owner", "session", "stopped")
    worker = PreparedWorker.resolve(
        family,
        WorkerScope("stopped", "stop-agent", "gpu-2", 0),
        bundle=bundle_reference,
    )
    with (
        HostArena(admission, 64 << 20, 100) as transfers,
        HostArena(admission, 16 << 20, 2) as commands,
        HostArena(admission, 64 << 20, 3) as publication,
        HostArena(admission, 160 << 20, 8) as storage_arena,
        HostArena(
            admission, execution.max_staging_bytes, execution.max_pending_requests
        ) as uploads,
        persistent_storage(storage_arena.budget, incarnation="stop-agent") as storage,
        storage.artifacts("owner") as artifacts,
        owned_node(transfers.budget, worker) as node,
    ):
        config = artifacts.get("configuration")
        storage.catalog.create(
            identity, request_sha256=config.sha256, configuration=config
        )
        storage.catalog.progress(identity, "initializing")
        processes = node.start(artifacts.open)
        with (
            start_processes(processes, stop=node.stop) as group,
            WorkerRequests(
                group, {0: worker}, {0: node.files}, metadata=commands.budget
            ) as transfer,
            storage.requests(identity) as journal,
            ResultPublisher(
                artifacts,
                session_id="session",
                history_id="stop-history",
                root=directory,
                memory=publication.budget,
                disk=disk,
            ) as publisher,
            RunController(
                journal,
                PreparedRequests(family, cache_root=directory),
                transfer,
                publisher,
                server_incarnation="stop-agent",
                budget=uploads.budget,
            ) as controller,
        ):
            storage.catalog.ready(
                identity, plans=bundle_reference, initial_history_id="stop-history"
            )
            controller.attach("writer")
            with snapshot_batch(records(targets=True), publication.budget) as snapshot:
                payload = ArtifactRef(
                    "backward", snapshot.sha256, snapshot.size_bytes, MEDIA_TYPE
                )
                wire = b"".join(snapshot.chunks())
                command = OperationRequest(
                    RequestIdentity(
                        "backward", "stopped", 1, "writer", 1, "stop-agent"
                    ),
                    "forward_backward",
                    payload,
                    len(records(targets=True)),
                    loss_params={"temperature": 2.0},
                )
            controller.register(command)
            controller.receive("backward", io.BytesIO(wire))
            result = controller.wait("backward", timeout=120)
            controller.drain(timeout=120)
            assert (
                result.result.accumulation.open and result.result.completed_updates == 0
            )
            # Allocation ownership can end with unresolved gradients. It must
            # not synthesize an optimizer update, a discard, or a valid new run.
            controller.stop(node.stop, timeout=30)
            assert controller.closed and not controller.boundary.valid
            assert controller.boundary.accumulation == result.result.accumulation
            assert controller.boundary.completed_updates == 0
            assert controller.register(command) == result
            assert processes[0].child.closed and node.lease.closed
            storage.catalog.stop(
                identity, ServiceError("allocation_stopped", "Allocation ended.")
            )
        node.close()
    assert admission.used.claims == 0 and disk.used == (0, 0)
    (directory / "allocation-stop.json").write_text(
        json.dumps(
            {
                "completed_updates": 0,
                "last_known_accumulation": result.result.accumulation.to_dict(),
                "live_state_valid": False,
                "worker_exit_confirmed": True,
                "all_claims_released": True,
            },
            indent=2,
        )
        + "\n"
    )


qualify_allocation_stop()
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
print("Qualified durable GPU artifacts after node and output-file cleanup")
print("Qualified production caller-order result and scalar publication")
print("Qualified ordered production discard followed by fresh backward/update")
print("Qualified ordered close with confirmed child exit and queryable result")
print("Qualified authenticated HTTP queue and post-exit artifact downloads")
print("Qualified bounded service storage and original GPU outcomes after restart")
print("Qualified allocation stop with unresolved GPU gradients and no implicit update")
