"""Actual scoped stores reopen and classify loss before serving saved requests."""

import hashlib
import io
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import replace
from threading import Thread

import pytest

from shadowspill.server import storage as storage_module
from shadowspill.server.artifacts import INDEX_BYTES as ARTIFACT_BYTES
from shadowspill.server.authentication import BearerTokens
from shadowspill.server.http import HTTP_SCRATCH_BYTES, HTTPService, RunAccess
from shadowspill.server.records.requests import RunIdentity
from shadowspill.server.records.runs import INDEX_BYTES as CATALOG_BYTES
from shadowspill.server.storage import ServiceStorage
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.requests import (
    ArtifactRef,
    OperationRequest,
    RequestIdentity,
)
from shadowspill_client.protocol.results import ExecutionReceipt, OperationResult
from shadowspill_client.transport.http import HTTPTransport, RemoteError

IDENTITY = RunIdentity("owner", "session", "run")
CONFIG = ArtifactRef("config", "a" * 64, 100)
PLANS = ArtifactRef("plans", "b" * 64, 1000)
HOST_BYTES = 128 << 20


def storage(root, *, incarnation="server", memory=None, **options):
    return ServiceStorage(
        root / "service",
        cache_root=root / "cache",
        incarnation=incarnation,
        memory=memory or BufferBudget(1 << 30, 16),
        request_host_bytes=HOST_BYTES,
        max_request_owners=1,
        max_artifact_owners=1,
        artifact_bytes=4 << 20,
        artifact_count=100,
        **options,
    )


def create(service, identity=IDENTITY):
    return service.catalog.create(
        identity, request_sha256="c" * 64, configuration=CONFIG
    )


def ready(service, identity=IDENTITY):
    service.catalog.progress(identity, "initializing")
    return service.catalog.ready(identity, plans=PLANS, initial_history_id="history")


def command(number, *, identity=IDENTITY):
    return OperationRequest(
        RequestIdentity(str(number), identity.run_id, number, "writer", 1, "server"),
        "step",
    )


def result(request):
    return OperationResult(
        request.identity, "step", "history", "completed", 1, grad_scale=1
    )


def add_history(records):
    records.claim_writer("writer", "server")
    for index in range(1, 6):
        request = command(index)
        records.submit(request)
        if index > 1:
            records.accept(str(index))
        if index > 2:
            records.dispatch(str(index))
        if index > 3:
            records.executed(ExecutionReceipt.from_result(result(request)))
        if index > 4:
            records.complete(result(request))


def test_startup_classifies_saved_effects_without_recreating_workers_or_history(
    tmp_path,
):
    unstarted = replace(IDENTITY, run_id="unstarted")
    with storage(tmp_path) as service:
        with pytest.raises(RuntimeError, match="classify"), service.requests(IDENTITY):
            pass
        service.recover()
        create(service, unstarted)
        create(service)
        with service.requests(IDENTITY) as records:
            add_history(records)
        original = ready(service)
    memory = BufferBudget(1 << 30, 16)
    with storage(tmp_path, incarnation="new", memory=memory) as service:
        service.recover()
        service.recover()
        info = service.catalog.get(IDENTITY)
        assert service.recovered and not service.catalog.interrupted()
        assert info.state == "stopped" and info.server_incarnation == "server"
        assert info.plans == original.plans and info.initial_history_id == "history"
        assert service.catalog.get(unstarted).state == "stopped"
        assert not service._request_path(unstarted).exists()
        assert create(service) == (info, False)
        with service.requests(IDENTITY) as records:
            assert not records.pending()
            effects = [records.get(str(n))[1].failure for n in range(1, 5)]
            assert [f.execution_effect for f in effects] == [
                "not_started",
                "not_started",
                "unknown",
                "completed",
            ]
            assert all(
                not f.state_valid and f.code == "server_restarted" for f in effects
            )
            assert effects[-1].completed_updates == 1
            assert records.get("5")[1].result == result(command(5))
            assert records.submit(command(4)) == records.get("4")[1]
    assert memory.used == (0, 0) and not list((tmp_path / "cache").iterdir())


def test_same_ids_in_different_owners_sessions_and_runs_have_distinct_storage(tmp_path):
    identities = (
        IDENTITY,
        replace(IDENTITY, owner_id="other"),
        replace(IDENTITY, session_id="other"),
        replace(IDENTITY, run_id="other"),
    )
    memory = BufferBudget(1 << 30, 16)
    with storage(tmp_path, memory=memory) as service:
        service.recover()
        for index, identity in enumerate(identities):
            create(service, identity)
            with service.requests(identity) as records:
                records.claim_writer(f"writer-{index}", "server")
            assert memory.used == (CATALOG_BYTES + HOST_BYTES, 2)
        assert len({service._request_path(i) for i in identities}) == 4
        for index, identity in enumerate(identities):
            with service.requests(identity) as records:
                assert records.writer.writer_id == f"writer-{index}"
        paths = set((tmp_path / "service" / "requests").iterdir())
        with (
            pytest.raises(KeyError),
            service.requests(replace(IDENTITY, run_id="missing")),
        ):
            pass
        assert set((tmp_path / "service" / "requests").iterdir()) == paths
    assert memory.used == (0, 0)


def test_artifact_eviction_preserves_payloads_and_authenticated_namespace(tmp_path):
    memory = BufferBudget(1 << 30, 16)
    references = {}
    with storage(tmp_path, memory=memory) as service:
        service.recover()
        for owner in ("owner", "other"):
            data = owner.encode()
            reference = ArtifactRef(
                "same-id", hashlib.sha256(data).hexdigest(), len(data)
            )
            references[owner] = reference
            with service.artifacts(owner) as artifacts:
                artifacts.put(reference, lambda _, data=data: io.BytesIO(data))
                assert memory.used == (CATALOG_BYTES + ARTIFACT_BYTES, 2)
        for owner, reference in references.items():
            with (
                service.artifacts(owner) as artifacts,
                artifacts.open(reference) as stream,
            ):
                assert stream.read() == owner.encode()
                assert artifacts.get("same-id") == reference
                with (
                    pytest.raises(KeyError),
                    artifacts.open(
                        references["other" if owner == "owner" else "owner"]
                    ),
                ):
                    pass
    assert memory.used == (0, 0)
    with storage(tmp_path, incarnation="new", memory=memory) as service:
        service.recover()
        with service.artifacts("owner") as artifacts:
            assert artifacts.get("same-id") == references["owner"]


def test_shutdown_retains_global_lock_and_host_claims_until_every_borrow_finishes(
    tmp_path,
):
    memory = BufferBudget(1 << 30, 16)
    service = storage(tmp_path, memory=memory)
    service.recover()
    create(service)
    other = replace(IDENTITY, run_id="other")
    create(service, other)
    with service.requests(IDENTITY) as records, service.artifacts("owner"):
        records.claim_writer("writer", "server")
        with pytest.raises(TimeoutError), service.requests(other):
            pass
        with pytest.raises(TimeoutError), service.artifacts("other"):
            pass
        with pytest.raises(ExceptionGroup):
            service.close()
        assert memory.used == (CATALOG_BYTES + HOST_BYTES + ARTIFACT_BYTES, 3)
        with pytest.raises(RuntimeError, match="active writer"):
            storage(tmp_path, incarnation="new", memory=memory)
        with pytest.raises(RuntimeError, match="closing"), service.requests(IDENTITY):
            pass
        assert records.writer.writer_id == "writer"
    service.close()
    service.close()
    assert memory.used == (0, 0) and service.closed
    with storage(tmp_path, incarnation="new", memory=memory) as reopened:
        reopened.recover()


def test_host_reservation_precedes_journal_construction(tmp_path, monkeypatch):
    memory = BufferBudget(CATALOG_BYTES + HOST_BYTES, 3)
    with storage(tmp_path, memory=memory) as service:
        service.recover()
        create(service)

        def forbidden(*args, **kwargs):
            raise AssertionError("journal opened without a host reservation")

        monkeypatch.setattr(storage_module, "RequestRecords", forbidden)
        with memory.reserve(1), pytest.raises(TimeoutError), service.requests(IDENTITY):
            pass
        assert memory.used == (CATALOG_BYTES, 1)
        assert not service._request_path(IDENTITY).exists()


@pytest.mark.parametrize("empty", [False, True])
def test_missing_initialized_history_is_an_error_not_a_new_empty_run(tmp_path, empty):
    with storage(tmp_path) as service:
        service.recover()
        create(service)
        ready(service)
        path = service._request_path(IDENTITY)
        if empty:
            path.mkdir()
    with storage(tmp_path, incarnation="new") as service:
        with pytest.raises(ValueError if empty else FileNotFoundError):
            service.recover()
        assert not service.recovered and service.catalog.get(IDENTITY).state == "ready"
        assert not list(path.glob("*.ssj"))


def test_failed_cleanup_keeps_the_catalog_lock_and_request_bytes_for_retry(
    tmp_path, monkeypatch
):
    memory = BufferBudget(1 << 30, 16)
    service = storage(tmp_path, memory=memory)
    service.recover()
    create(service)
    with service.requests(IDENTITY) as records:
        cleanup = records._index._directory.cleanup

    def fail():
        raise OSError("local index cleanup failed")

    monkeypatch.setattr(records._index._directory, "cleanup", fail)
    with pytest.raises(ExceptionGroup):
        service.close()
    assert memory.used == (CATALOG_BYTES + HOST_BYTES, 2) and not service.closed
    with pytest.raises(RuntimeError, match="active writer"):
        storage(tmp_path, incarnation="new", memory=memory)
    monkeypatch.setattr(records._index._directory, "cleanup", cleanup)
    service.close()
    assert memory.used == (0, 0)


def test_interrupted_recovery_can_reopen_without_duplicating_terminal_effects(
    tmp_path, monkeypatch
):
    with storage(tmp_path) as service:
        service.recover()
        create(service)
        with service.requests(IDENTITY) as records:
            add_history(records)
        ready(service)
    with storage(tmp_path, incarnation="second") as service:

        def fail(*args, **kwargs):
            raise OSError("catalog stop was not acknowledged")

        monkeypatch.setattr(service.catalog, "stop", fail)
        with pytest.raises(OSError):
            service.recover()
        assert not service.recovered
    with storage(tmp_path, incarnation="third") as service:
        service.recover()
        assert service.catalog.get(IDENTITY).state == "stopped"
        with service.requests(IDENTITY) as records:
            assert records.get("3")[1].failure.execution_effect == "unknown"
            assert records.get("4")[1].failure.execution_effect == "completed"
            assert records.get("5")[1].result == result(command(5))


def test_reopened_storage_serves_original_results_and_artifacts_over_actual_http(
    tmp_path,
):
    data = b"immutable result"
    reference = ArtifactRef("output", hashlib.sha256(data).hexdigest(), len(data))
    with storage(tmp_path) as service:
        service.recover()
        create(service)
        with service.requests(IDENTITY) as records:
            add_history(records)
        ready(service)
        with service.artifacts("owner") as artifacts:
            artifacts.put(reference, lambda _: io.BytesIO(data))
    with storage(tmp_path, incarnation="new") as service:
        service.recover()

        @contextmanager
        def lookup(identity):
            with service.requests(identity) as records:
                yield RunAccess(records, None)

        token = "a" * 32
        server_memory = BufferBudget(2 * HTTP_SCRATCH_BYTES, 2)
        client_memory = BufferBudget(HTTP_SCRATCH_BYTES, 1)
        http = HTTPService(
            ("127.0.0.1", 0),
            incarnation="new",
            authentication=BearerTokens({token: "owner"}),
            runs=lookup,
            artifacts=service.artifacts,
            memory=server_memory,
        )
        thread = Thread(target=http.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        client = HTTPTransport(
            f"http://127.0.0.1:{http.server_port}",
            token=token,
            memory=client_memory,
            timeout=5,
        )
        try:
            saved = client.request("GET", "/v1/runs/session/run/requests/5").value
            assert saved.result == result(command(5))
            assert (
                client.request(
                    "POST", "/v1/runs/session/run/requests", command(5)
                ).value
                == saved
            )
            with pytest.raises(RemoteError):
                client.request("POST", "/v1/runs/session/run/requests", command(6))
            target = io.BytesIO()
            client.download("/v1/artifacts/output", reference, target)
            assert target.getvalue() == data
        finally:
            http.shutdown()
            http.server_close()
            thread.join(5)
        assert not thread.is_alive()
        assert server_memory.used == client_memory.used == (0, 0)


def test_service_storage_recovery_does_not_import_gpu_or_catalog(tmp_path):
    child = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import importlib.abc, sys
from pathlib import Path
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, *args):
        if name.split('.')[0] in {'torch', 'triton', 'pynvml', 'shadowspill_workloads'}:
            raise AssertionError('framework in service storage: ' + name)
sys.meta_path.insert(0, Block())
from shadowspill.server.storage import ServiceStorage
from shadowspill_client.buffers import BufferBudget
root = Path(sys.argv[1])
memory = BufferBudget(256 << 20, 4)
with ServiceStorage(root/'service', cache_root=root/'cache', incarnation='server',
    memory=memory, request_host_bytes=128 << 20, max_request_owners=1,
    max_artifact_owners=1, artifact_bytes=1024, artifact_count=10) as storage:
    storage.recover()
    assert storage.recovered
assert memory.used == (0, 0)
""",
            str(tmp_path),
        ],
        capture_output=True,
        timeout=15,
    )
    assert child.returncode == 0, child.stderr.decode()
