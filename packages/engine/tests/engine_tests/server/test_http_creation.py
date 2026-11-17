"""Creation HTTP routes retain existing factory, catalog and lifetime ownership."""

from contextlib import contextmanager
from dataclasses import replace
from threading import Event, Thread

import pytest

from engine_tests.server.test_http import OTHER_TOKEN, ROOT, TOKEN
from engine_tests.server.test_jobs import until
from engine_tests.server.test_local_creation import configured
from engine_tests.server.test_storage import IDENTITY
from engine_tests.training.test_setup_resolution import arguments
from shadowspill.server.authentication import BearerTokens
from shadowspill.server.http import HTTP_SCRATCH_BYTES, HTTPService
from shadowspill.server.records.requests import RunIdentity
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.creation import TrainingCreation
from shadowspill_client.protocol.requests import OperationRequest, RequestIdentity
from shadowspill_client.protocol.service import ServiceError
from shadowspill_client.protocol.setup import TrainingSetup
from shadowspill_client.transport.http import HTTPTransport, RemoteError


@contextmanager
def endpoint(ctx):
    memory = BufferBudget(HTTP_SCRATCH_BYTES * 2, 2)
    ctx.http_memory = memory
    client_memory = BufferBudget(HTTP_SCRATCH_BYTES * 2, 2)

    def describe(settings):
        with ctx.job_memory.reserve(ctx.jobs.host_bytes, timeout=0):
            return ctx.factory.describe(settings)

    service = HTTPService(
        ("127.0.0.1", 0),
        incarnation="server",
        authentication=BearerTokens({TOKEN: "owner", OTHER_TOKEN: "other"}),
        runs=ctx.jobs.access,
        artifacts=ctx.service.artifacts,
        jobs=ctx.jobs,
        describe=describe,
        memory=memory,
        request_timeout=5,
    )
    thread = Thread(target=service.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    base = f"http://127.0.0.1:{service.server_port}"
    client = HTTPTransport(base, token=TOKEN, memory=client_memory, timeout=5)
    other = HTTPTransport(base, token=OTHER_TOKEN, memory=client_memory, timeout=5)
    try:
        yield client, other
    finally:
        service.shutdown()
        service.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert memory.used == client_memory.used == (0, 0)


def test_http_creation_waits_for_creator_exit_then_uses_existing_ordered_close(
    tmp_path, monkeypatch
):
    initialized, release = Event(), Event()

    def fault(owner):
        original = owner.start

        def start():
            original()
            initialized.set()
            assert release.wait(5)

        monkeypatch.setattr(owner, "start", start)

    with (
        configured(tmp_path, monkeypatch, fault=fault) as ctx,
        endpoint(ctx) as (
            client,
            other,
        ),
    ):
        creation = TrainingCreation(ctx.config, name="demo", labels={"job": "test"})
        try:
            reply = client.request("POST", ROOT, creation)
            assert reply.status == 202 and reply.value.name == "demo"
            assert initialized.wait(5)
            waiting = client.request("GET", ROOT)
            assert waiting.status == 202 and waiting.value.state == "ready"
            assert not ctx.jobs.available(IDENTITY)
            assert client.request("POST", ROOT, creation).value == waiting.value
            assert len(ctx.owners) == 1  # Exact retry did not run another factory.
            with pytest.raises(RemoteError) as rejected:
                client.request("POST", ROOT, replace(creation, name="different"))
            assert rejected.value.status == 400
            with pytest.raises(RemoteError) as denied:
                other.request("GET", ROOT)
            assert denied.value.status == 404
            assert other.request("GET", "/v1/runs/session").value.runs == ()
            with ctx.service.artifacts("owner") as artifacts:
                before = artifacts.page()
                with pytest.raises(RemoteError) as full:
                    client.request(
                        "POST",
                        "/v1/runs/session/second",
                        replace(creation, name="next"),
                    )
                assert full.value.status == 429 and artifacts.page() == before
            assert (
                ctx.service.catalog.get(RunIdentity("owner", "session", "second"))
                is None
            )
        finally:
            release.set()
        until(lambda: client.request("GET", ROOT).status == 200)
        assert ctx.jobs.available(IDENTITY)
        lease = client.request("POST", ROOT + "/writers/writer").value
        close = OperationRequest(
            RequestIdentity("close", "run", 1, "writer", lease.writer_epoch, "server"),
            "close",
        )
        client.request("POST", ROOT + "/requests", close)
        client.request("PUT", ROOT + "/requests/close/input")
        until(lambda: client.request("GET", ROOT).value.state == "closed")
        assert client.request("POST", ROOT, creation).value.state == "closed"
        until(lambda: ctx.jobs.used == (0, 0))
        assert len(ctx.owners) == 1 and ctx.jobs.used == (0, 0)
        assert (
            client.request("GET", ROOT + "/requests/close").value.state == "completed"
        )
        page = client.request("GET", "/v1/runs/session").value
        assert [run.run_id for run in page.runs] == ["run"]
        assert page.next_cursor is None


def test_http_failed_creation_is_queryable_without_initialization_or_replay(
    tmp_path, monkeypatch
):
    with (
        configured(
            tmp_path,
            monkeypatch,
            change=lambda c: replace(c, model=replace(c.model, initialize=None)),
        ) as ctx,
        endpoint(ctx) as (client, _),
    ):
        creation = TrainingCreation(ctx.config)
        client.request("POST", ROOT, creation)
        until(lambda: client.request("GET", ROOT).value.state == "failed")
        result = client.request("GET", ROOT)
        assert result.status == 200 and result.value.error.code == "creation_failed"
        assert client.request("POST", ROOT, creation).value == result.value
        assert not ctx.phases and len(ctx.owners) == 1


def test_http_run_pages_fit_the_control_envelope_without_losing_cursor_rows(
    tmp_path, monkeypatch
):
    with configured(tmp_path, monkeypatch) as ctx, endpoint(ctx) as (client, other):
        from shadowspill.server.startup.records import CreationRecord

        creation = CreationRecord.encode(TrainingCreation(ctx.config))
        labels = {f"label-{i}": "x" * 256 for i in range(16)}
        ids = [f"run-{i:03}" for i in range(128)]
        for run_id in ids:
            identity = RunIdentity("owner", "session", run_id)
            ctx.service.catalog.create(
                identity,
                request_sha256=creation.reference.sha256,
                configuration=creation.configuration,
                labels=labels,
            )
            ctx.service.catalog.stop(
                identity, ServiceError("stopped", "\U0010ffff" * 1024)
            )
        seen, cursor = [], ""
        while True:
            page = client.request(
                "GET", f"/v1/runs/session?limit=128&after={cursor}"
            ).value
            assert len(page.runs) <= 32
            seen.extend(run.run_id for run in page.runs)
            if page.next_cursor is None:
                break
            assert page.next_cursor != cursor
            cursor = page.next_cursor
        assert seen == ids
        assert other.request("GET", "/v1/runs/session?limit=128").value.runs == ()
        for query in (
            "limit=129",
            "limit=0",
            "limit=-1",
            "unknown=x",
            "limit=2&limit=3",
        ):
            with pytest.raises(RemoteError) as bad:
                client.request("GET", "/v1/runs/session?" + query)
            assert bad.value.status == 400


def test_oversize_registration_publishes_no_artifacts_or_creation_job(
    tmp_path, monkeypatch
):
    with configured(tmp_path, monkeypatch) as ctx, endpoint(ctx) as (client, _):
        creation = TrainingCreation(
            ctx.config, labels={str(i): "界" * 256 for i in range(16)}
        )
        with ctx.service.artifacts("owner") as artifacts:
            assert artifacts.page() == ()
            with pytest.raises(RemoteError) as invalid:
                client.request("POST", ROOT, creation)
            assert invalid.value.status == 400 and artifacts.page() == ()
        assert ctx.jobs.used == (0, 0) and not ctx.owners
        assert ctx.service.catalog.get(IDENTITY) is None


def test_http_setup_inspection_is_framework_free_and_creates_no_worker_or_records(
    tmp_path, monkeypatch
):
    with configured(tmp_path, monkeypatch) as ctx, endpoint(ctx) as (client, _):
        values = arguments()
        values["model"] = replace(values["model"], initialize=None)
        before = ctx.node.used
        preview = client.request(
            "POST", "/v1/training/describe", TrainingSetup(**values)
        ).value
        config = preview.config
        assert config.model.initialize is None
        assert config.objective.defaults == {"temperature": 1.0}
        assert client.request("POST", "/v1/training/describe", config).value == preview
        assert ctx.node.used == before and ctx.job_memory.used == (0, 0)
        assert ctx.jobs.used == (0, 0) and not ctx.owners and not ctx.phases
        assert ctx.service.catalog.page("owner", "session") == ()
        with ctx.service.artifacts("owner") as artifacts:
            assert artifacts.page() == ()
        with pytest.raises(RemoteError) as bad:
            client.request("POST", "/v1/training/describe", TrainingCreation(config))
        assert bad.value.status == 400
        with ctx.job_memory.reserve(ctx.jobs.host_bytes, timeout=0):
            with pytest.raises(RemoteError) as busy:
                client.request("POST", "/v1/training/describe", config)
            assert busy.value.status == 429
