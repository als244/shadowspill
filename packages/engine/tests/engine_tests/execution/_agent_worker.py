"""Framework-free controller, actual installed GPU child and owned binary files."""

import hashlib
import importlib.abc
import io
import json
import os
import shlex
import subprocess
import sys
import tarfile
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Thread
from time import monotonic
from types import SimpleNamespace


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
from shadowspill.server import Allocation, Server, ServerLimits  # noqa: E402
from shadowspill.server.authentication import BearerTokens  # noqa: E402
from shadowspill.server.http import HTTP_SCRATCH_BYTES, HTTPService, RunAccess  # noqa: E402
from shadowspill.server.jobs import RunJobs  # noqa: E402
from shadowspill.server.live import LiveRun  # noqa: E402
from shadowspill.server.records.requests import RunIdentity  # noqa: E402
from shadowspill.server.results import ResultPublisher  # noqa: E402
from shadowspill.server.startup.configuration import LocalCreationLimits  # noqa: E402
from shadowspill.server.storage import ServiceStorage  # noqa: E402
from shadowspill.training.definitions import load_registry  # noqa: E402
from shadowspill.training.preparation.artifacts.bundle import TrainingBundle  # noqa: E402
from shadowspill.training.preparation.phases.job import (  # noqa: E402
    JOB_NAME,
    PreparationJob,
)
from shadowspill.training.session.controller import RunController  # noqa: E402
from shadowspill.training.session.prepared import PreparedRequests  # noqa: E402
from shadowspill.training.session.state import OperationState  # noqa: E402
from shadowspill.training.worker.launch import start_processes  # noqa: E402
from shadowspill.training.worker.node import WorkerNode  # noqa: E402
from shadowspill.training.worker.protocol import WorkerScope  # noqa: E402
from shadowspill.training.worker.requests import WorkerRequests  # noqa: E402
from shadowspill.training.worker.resources import WorkerHost  # noqa: E402
from shadowspill.training.worker.runtime import PreparedWorker  # noqa: E402
from shadowspill_client.buffers import BufferBudget  # noqa: E402
from shadowspill_client.protocol.codec import decode, encode  # noqa: E402
from shadowspill_client.protocol.payload import MEDIA_TYPE, snapshot_batch  # noqa: E402
from shadowspill_client.protocol.payload_io import read_payload  # noqa: E402
from shadowspill_client.protocol.requests import (  # noqa: E402
    ArtifactRef,
    OperationRequest,
    RequestIdentity,
)
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
        artifact_bytes=256 << 20,
        artifact_count=400,
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
            state=OperationState(history_id="history"),
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
        HostArena(admission, 128 << 20, 1) as job_arena,
        HostArena(
            admission, execution.max_staging_bytes, execution.max_pending_requests
        ) as uploads,
        persistent_storage(storage_arena.budget, incarnation="stop-agent") as storage,
        storage.artifacts("owner") as artifacts,
        owned_node(transfers.budget, worker) as node,
    ):
        config = artifacts.get("configuration")
        live = LiveRun(
            storage,
            identity,
            family,
            [node],
            plans=bundle_reference,
            history_id="stop-history",
            root=directory,
            metadata=commands.budget,
            uploads=uploads.budget,
            publication=publication.budget,
            disk=disk,
        )
        owner = SimpleNamespace(
            run=live,
            start=live.start,
            interrupt=live.interrupt,
            close=live.close,
        )
        jobs = RunJobs(
            storage,
            lambda _: owner,
            memory=job_arena.budget,
            host_bytes=128 << 20,
            max_runs=1,
            reap_interval=0.01,
        )
        jobs.start()
        try:
            jobs.create(identity, request_sha256=config.sha256, configuration=config)
            deadline = monotonic() + 120
            while not next(iter(jobs._jobs.values())).live:
                assert monotonic() < deadline, jobs.failures()
                assert storage.catalog.get(identity).state == "preparing" or live.ready
                Event().wait(0.01)
            controller = live.controller
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
            with jobs.access(identity) as access:
                assert access.live() is controller
                try:
                    jobs.close(timeout=1)
                except RuntimeError as error:
                    assert "still own" in str(error)
                else:
                    raise AssertionError(
                        "borrowed live run retired before HTTP returned"
                    )
                deadline = monotonic() + 30
                while not controller.closed:
                    assert monotonic() < deadline, jobs.failures()
                    Event().wait(0.01)
                assert not live.closed and not node.closed
                assert job_arena.budget.used == (128 << 20, 1)
            jobs.close(timeout=30)
            assert jobs.closed and job_arena.budget.used == (0, 0)
            assert controller.closed and not controller.boundary.valid
            assert controller.boundary.accumulation == result.result.accumulation
            assert controller.boundary.completed_updates == 0
            assert controller.register(command) == result
            assert node.closed and node.lease.closed and live.closed
            assert storage.catalog.get(identity).state == "stopped"
            with storage.requests(identity) as saved:
                assert saved.get("backward")[1] == result
            assert any(ref.id.startswith("worker-logs.") for ref in artifacts.page())
        finally:
            jobs.close(timeout=30)
            live.stop()
        node.close()
    assert admission.used.claims == 0 and disk.used == (0, 0)
    (directory / "allocation-stop.json").write_text(
        json.dumps(
            {
                "completed_updates": 0,
                "last_known_accumulation": result.result.accumulation.to_dict(),
                "live_state_valid": False,
                "worker_exit_confirmed": True,
                "job_host_bytes": 128 << 20,
                "borrow_retained_until_http_return": True,
                "all_claims_released": True,
            },
            indent=2,
        )
        + "\n"
    )


qualify_allocation_stop()


def qualify_public_server():
    receipts = []
    with Server(
        allocation=Allocation(
            {
                **os.environ,
                "TORCHINDUCTOR_CACHE_DIR": str(directory / "factory-inductor"),
                "TRITON_CACHE_DIR": str(directory / "factory-triton"),
            }
        ),
        service_root=directory / "public-service",
        cache_root=directory / "public-cache",
        workload_packages=packages,
        authentication=BearerTokens({token: "owner"}),
        limits=ServerLimits(
            max_http_requests=2,
            max_open_records=1,
            max_artifact_owners=1,
            artifact_bytes=256 << 20,
            artifact_count=400,
            temporary_disk_bytes=512 << 20,
            temporary_files=200,
            creation=LocalCreationLimits(
                preparation_host_bytes=8 << 30,
                preparation_disk_bytes=256 << 20,
                worker=host_allowance,
                transfer_bytes=64 << 20,
                publication_bytes=64 << 20,
                log_bytes=64 << 10,
                max_preparation_files=1024,
            ),
        ),
    ) as service:
        service.start(port=0)
        # Inspect internal owners only to qualify cleanup; all setup and training
        # use public APIs on both sides of the connection.
        jobs, storage = service._jobs, service._storage
        node, disk = service._node, service._disk
        with storage.artifacts("owner") as artifacts:
            options = {
                "port": service.address[1],
                "token": token,
                "config": prepared_worker.bindings.setup.config.to_dict(),
                "oracle": oracle,
            }
            host = os.environ.get("SHADOWSPILL_DEMO_CLIENT_HOST")
            if host:
                command = [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=10",
                    host,
                    shlex.join(
                        [
                            os.environ["SHADOWSPILL_DEMO_CLIENT_PYTHON"],
                            "-I",
                            os.environ["SHADOWSPILL_DEMO_CLIENT_SCRIPT"],
                        ]
                    ),
                ]
                options["ssh_host"] = "chicago"
            else:
                command = [
                    sys.executable,
                    str(Path(__file__).with_name("_service_client.py")),
                ]
            result = subprocess.run(
                command,
                input=json.dumps(options),
                capture_output=True,
                text=True,
                timeout=360,
            )
            assert result.returncode == 0, result.stdout + result.stderr
            client_receipt = json.loads(result.stdout)
            assert client_receipt["all_client_claims_released"]
            if host:
                assert client_receipt["client_host"] == "tubingen"
            (directory / "demo-client.json").write_text(
                json.dumps(client_receipt, indent=2) + "\n"
            )
            deadline = monotonic() + 30
            while jobs.used != (0, 0):
                assert monotonic() < deadline, jobs.failures()
                Event().wait(0.01)
            assert not jobs.failures() and disk.used == (0, 0)
            assert node.used.host_bytes == service.limits.fixed_host_bytes
            assert not node.used.devices
            for result in client_receipt["runs"]:
                mode = result["mode"]
                identity = RunIdentity("owner", "factory-session", result["run_id"])
                info = storage.catalog.get(identity)
                assert info.state == "closed"
                checkpoint_ref = ArtifactRef.from_dict(result["checkpoint"])
                resumed_identity = RunIdentity(
                    "owner", "factory-session", result["new_run_id"]
                )
                resumed_info = storage.catalog.get(resumed_identity)
                assert resumed_info.state == "closed"
                assert resumed_info.parent_checkpoint == checkpoint_ref
                assert resumed_info.initial_completed_updates == 1
                assert resumed_info.initial_history_id == result["new_run_history_id"]
                assert resumed_info.configuration == info.configuration
                assert result["new_run_resume_update_oracle"]
                conversion = result["conversion"]
                converted_info = storage.catalog.get(
                    RunIdentity("owner", "factory-session", conversion["run_id"])
                )
                assert converted_info.state == "closed"
                assert converted_info.parent_checkpoint == checkpoint_ref
                assert converted_info.initial_checkpoint == ArtifactRef.from_dict(
                    conversion["initial_checkpoint"]
                )
                assert converted_info.initial_checkpoint != checkpoint_ref
                assert converted_info.initial_completed_updates == 1
                assert (
                    converted_info.initial_history_id
                    == conversion["initial_history_id"]
                )
                assert converted_info.plans == ArtifactRef.from_dict(
                    conversion["plans"]
                )
                assert (
                    conversion["split_update_oracle"]
                    and conversion["whole_step_oracle"]
                )
                with artifacts.open(converted_info.initial_checkpoint) as stream:
                    converted_manifest = decode(stream.read())
                assert converted_manifest.info.conversion.source == checkpoint_ref
                assert converted_manifest.info.plans == converted_info.plans
                rng_conversion = converted_manifest.info.conversion.randomness
                assert rng_conversion["root_seed"] == execution.seed
                assert {row["action"] for row in rng_conversion["target_streams"]} == {
                    "preserve"
                }
                assert (
                    artifacts.resolve_alias(
                        "factory-session", result["run_id"], "step-1"
                    )
                    == checkpoint_ref
                )
                with artifacts.open(checkpoint_ref) as stream:
                    checkpoint_manifest = decode(stream.read())
                assert checkpoint_manifest.info.completed_updates == 1
                assert checkpoint_manifest.info.plans == info.plans
                assert checkpoint_manifest.info.weights == result["checkpoint_weights"]
                for row in checkpoint_manifest.ranks:
                    from shadowspill.checkpoint.archive import read_manifest
                    from shadowspill.checkpoint.random_streams import read_snapshot

                    with (
                        artifacts.open(row.archive) as stream,
                        tarfile.open(fileobj=stream, mode="r:") as archive,
                    ):
                        rank_manifest = read_manifest(archive)
                    assert (
                        rank_manifest["metadata"]["checkpoint"]
                        == checkpoint_manifest.info.rank_metadata()["checkpoint"]
                    )
                    assert rank_manifest["state"] == row.state_sha256
                    assert rank_manifest["tensors"]
                    seed, streams = read_snapshot(
                        rank_manifest["metadata"]["randomness"]
                    )
                    assert seed == execution.seed and len(streams) == 8
                    assert all(
                        state.stream.rank == row.rank and state.payload
                        for state in streams
                    )
                assert result["checkpoint_download_after_close"]
                assert result["post_restore_update_oracle"]
                with storage.requests(identity) as records:
                    request, status = records.get(result["restore_request_id"])
                    assert request.operation == "restore"
                    assert status.acceptance.checkpoint == checkpoint_ref
                    assert status.result.checkpoint == checkpoint_ref
                    assert (
                        status.result.completed_updates
                        == result["restored_updates"]
                        == 1
                    )
                    assert status.result.history_id == result["restored_history_id"]
                    assert (
                        status.result.history_id != checkpoint_manifest.info.history_id
                    )
                    assert (
                        records.execution(request.identity.request_id).history_id
                        == status.result.history_id
                    )
                manifests = []
                for ref in artifacts.page(limit=128):
                    if ref.id.startswith("preparation."):
                        with artifacts.open(ref) as stream:
                            manifest = json.load(stream)
                        if manifest["run_id"] == result["run_id"]:
                            manifests.append(manifest)
                phases = sorted(m["phase"] for m in manifests if "phase" in m)
                assert phases == (
                    ["build", "communicate", "plan", "profile"]
                    if mode == "fresh"
                    else []
                )
                receipts.append(
                    {
                        "mode": mode,
                        "completed_updates": result["completed_updates"],
                        "phases": phases,
                        "preparation_manifests": len(manifests),
                        "plans": info.plans.to_dict(),
                        "all_run_claims_released": True,
                    }
                )
    assert service.closed and node.used.claims == 0 and disk.used == (0, 0)
    (directory / "factory-creation.json").write_text(
        json.dumps(receipts, indent=2) + "\n"
    )


qualify_public_server()
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
print("Qualified configured fresh and saved creation through actual GPU retirement")

print("Qualified ordered GPU checkpoints and post-exit downloads from the demo client")
print("Qualified GPU restore, counter/history rewind and the next optimizer update")
