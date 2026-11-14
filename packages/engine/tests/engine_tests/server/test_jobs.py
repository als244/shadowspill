"""Bound ownership through actual thread exit, client borrows and cleanup faults."""

import io
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from threading import Event, Thread
from time import monotonic
from types import SimpleNamespace

import pytest

from engine_tests.server.test_live import configured as live_run
from engine_tests.server.test_storage import CONFIG, IDENTITY, PLANS, storage
from shadowspill.server.authentication import BearerTokens
from shadowspill.server.http import HTTP_SCRATCH_BYTES, HTTPService
from shadowspill.server.jobs import RunJobs
from shadowspill.server.records.requests import RunIdentity
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.requests import OperationRequest, RequestIdentity
from shadowspill_client.protocol.service import ServiceError
from shadowspill_client.transport.http import HTTPTransport


def until(predicate):
    deadline = monotonic() + 5
    while not predicate():
        assert monotonic() < deadline
        Event().wait(0.01)


class FakeRun:
    """Only ownership evidence; real controller/child coverage is below."""

    def __init__(self):
        self.ready = self.closed = self.done = False
        self.controller = SimpleNamespace(failure=None)

    @property
    def finished(self):
        return self.closed or self.done or self.controller.failure is not None


class Job:
    def __init__(self, service, info, *, blocked=False, failed=False):
        self.service, self.info = service, info
        self.identity = RunIdentity(info.owner_id, info.session_id, info.run_id)
        self.run = FakeRun()
        self.entered, self.release = Event(), Event()
        if not blocked:
            self.release.set()
        self.failed = failed
        self.closing_error = self.stopping = self.closed = False
        self.starts = self.interrupts = 0
        self.decoded = bytearray(32 << 10)

    def start(self):
        self.starts += 1
        self.entered.set()
        assert self.release.wait(5)
        if self.stopping or self.failed:
            raise RuntimeError("stopped or failed preparation")
        with self.service.requests(self.identity):
            pass
        self.service.catalog.progress(self.identity, "initializing")
        self.service.catalog.ready(
            self.identity, plans=PLANS, initial_history_id="history"
        )
        self.run.ready = True

    def interrupt(self):
        self.interrupts += 1
        self.stopping = True
        self.release.set()

    def close(self):
        if self.closing_error:
            raise OSError("cleanup is blocked")
        if self.run.ready and not self.closed:
            if self.stopping:
                self.service.catalog.stop(
                    self.identity, ServiceError("allocation_stopped", "Stopped.")
                )
            else:
                self.service.catalog.finish(self.identity)
        self.run.closed = self.closed = True
        self.decoded = None


@contextmanager
def configured(tmp_path, *, max_runs=1, blocked=False, failed=False):
    with storage(tmp_path) as service:
        service.recover()
        made = []
        memory = BufferBudget(256 << 10, 4)

        def factory(info):
            job = Job(service, info, blocked=blocked, failed=failed)
            made.append(job)
            return job

        jobs = RunJobs(
            service,
            factory,
            memory=memory,
            host_bytes=64 << 10,
            max_runs=max_runs,
            reap_interval=0.01,
        )
        jobs.start()
        try:
            yield SimpleNamespace(jobs=jobs, made=made, service=service, memory=memory)
        finally:
            for job in made:
                job.closing_error = False
                job.release.set()
            jobs.close(timeout=5)
            assert jobs.closed and memory.used == (0, 0)


def create(jobs, identity=IDENTITY):
    return jobs.create(identity, request_sha256="c" * 64, configuration=CONFIG)


def live(jobs):
    until(lambda: next(iter(jobs._jobs.values())).live)


def test_capacity_precedes_registration_and_exact_retries_never_recreate(tmp_path):
    with configured(tmp_path, blocked=True) as ctx:
        original = create(ctx.jobs)
        job = ctx.made[0]
        assert job.entered.wait(5)
        assert create(ctx.jobs) == original and job.starts == 1
        other = replace(IDENTITY, run_id="other")
        with pytest.raises(TimeoutError, match="slots"):
            create(ctx.jobs, other)
        assert ctx.service.catalog.get(other) is None and len(ctx.made) == 1
        with pytest.raises(ValueError, match="different creation"):
            ctx.jobs.create(IDENTITY, request_sha256="d" * 64, configuration=CONFIG)
        job.release.set()
        live(ctx.jobs)
        # Maintenance does not turn idle/disconnected ownership into cancellation.
        Event().wait(0.05)
        assert not job.interrupts and not job.closed
        job.run.done = True
        until(lambda: ctx.jobs.used == (0, 0))
        saved = create(ctx.jobs)
        assert saved.state == "closed" and len(ctx.made) == 1 and job.starts == 1
        assert ctx.memory.used == (0, 0) and job.decoded is None


def test_host_memory_precedes_catalog_and_factory(tmp_path):
    with configured(tmp_path) as ctx, ctx.memory.reserve(ctx.memory.max_bytes):
        with pytest.raises(TimeoutError, match="staging capacity"):
            create(ctx.jobs)
        assert ctx.service.catalog.get(IDENTITY) is None and not ctx.made


def test_publication_obeys_admission_and_runs_once_before_registration(tmp_path):
    with configured(tmp_path, blocked=True) as ctx:
        published = []

        def publish():
            assert ctx.jobs.used == (1, 0) and ctx.memory.used == (64 << 10, 1)
            assert ctx.service.catalog.get(IDENTITY) is None and not ctx.made
            published.append(True)

        kwargs = dict(request_sha256="c" * 64, configuration=CONFIG, publish=publish)
        original = ctx.jobs.create(IDENTITY, **kwargs)
        assert ctx.jobs.create(IDENTITY, **kwargs) == original and published == [True]
        with pytest.raises(TimeoutError):
            ctx.jobs.create(replace(IDENTITY, run_id="other"), **kwargs)
        assert published == [True]


def test_failed_publication_has_no_registered_job_or_stranded_reservation(tmp_path):
    with configured(tmp_path) as ctx:

        def failed():
            raise OSError("metadata publication failed")

        with pytest.raises(OSError, match="publication"):
            ctx.jobs.create(
                IDENTITY, request_sha256="c" * 64, configuration=CONFIG, publish=failed
            )
        assert not ctx.made and ctx.jobs.used == ctx.memory.used == (0, 0)
        assert ctx.service.catalog.get(IDENTITY) is None
        assert create(ctx.jobs).state == "preparing"


def test_shutdown_and_duplicate_publication_wait_for_the_original_io(tmp_path):
    with configured(tmp_path) as ctx:
        entered, release = Event(), Event()

        def publish():
            entered.set()
            assert release.wait(5)

        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(
                ctx.jobs.create,
                IDENTITY,
                request_sha256="c" * 64,
                configuration=CONFIG,
                publish=publish,
            )
            assert entered.wait(5)
            try:
                with pytest.raises(TimeoutError, match="already being published"):
                    create(ctx.jobs)
                with pytest.raises(RuntimeError, match="still own"):
                    ctx.jobs.close(timeout=0.01)
                assert ctx.jobs.used == (1, 0) and ctx.memory.used == (64 << 10, 1)
                assert not ctx.made and ctx.service.catalog.get(IDENTITY) is None
            finally:
                release.set()
            future.result(timeout=5)
        ctx.jobs.close(timeout=5)
        assert ctx.service.catalog.get(IDENTITY).state == "stopped"
        assert ctx.made[0].starts == 0


def test_failed_creation_is_terminal_but_cleanup_retains_its_slot_and_envelope(
    tmp_path,
):
    with configured(tmp_path, blocked=True, failed=True) as ctx:
        create(ctx.jobs)
        job = ctx.made[0]
        job.closing_error = True
        job.release.set()
        until(ctx.jobs.failures)
        assert ctx.jobs.used == (1, 0) and ctx.memory.used == (64 << 10, 1)
        assert job.decoded is not None
        with pytest.raises(TimeoutError):
            create(ctx.jobs, replace(IDENTITY, run_id="other"))
        job.closing_error = False
        until(lambda: ctx.jobs.used == (0, 0))
        info = create(ctx.jobs)
        assert info.state == "failed" and info.error.code == "creation_failed"
        assert ctx.memory.used == (0, 0) and job.starts == 1


@pytest.mark.parametrize("where", ["factory", "thread"])
def test_failure_before_creation_thread_records_a_single_failed_identity(
    tmp_path, monkeypatch, where
):
    with configured(tmp_path) as ctx:

        def failed(*args):
            raise OSError("failed before start")

        if where == "factory":
            monkeypatch.setattr(ctx.jobs, "_factory", failed)
        else:
            monkeypatch.setattr(Thread, "start", failed)
        create(ctx.jobs)
        until(lambda: ctx.jobs.used == (0, 0))
        assert create(ctx.jobs).state == "failed"
        assert len(ctx.made) == (1 if where == "thread" else 0)
        assert not ctx.made or ctx.made[0].starts == 0


def test_shutdown_during_factory_keeps_admission_until_construction_returns(
    tmp_path, monkeypatch
):
    with configured(tmp_path) as ctx:
        entered, release = Event(), Event()
        original = ctx.jobs._factory

        def factory(info):
            entered.set()
            assert release.wait(5)
            return original(info)

        monkeypatch.setattr(ctx.jobs, "_factory", factory)
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(create, ctx.jobs)
            assert entered.wait(5)
            try:
                with pytest.raises(RuntimeError, match="still own"):
                    ctx.jobs.close(timeout=0.01)
                assert ctx.memory.used == (64 << 10, 1)
            finally:
                release.set()
            assert future.result(timeout=5).state == "preparing"
        ctx.jobs.close(timeout=5)
        assert ctx.made[0].starts == 0 and ctx.made[0].closed
        assert ctx.service.catalog.get(IDENTITY).state == "stopped"


def test_creation_event_does_not_release_a_thread_that_has_not_returned(
    tmp_path, monkeypatch
):
    with configured(tmp_path) as ctx:
        returned, release = Event(), Event()
        original = ctx.jobs._create

        def target(job):
            original(job)
            returned.set()
            assert release.wait(5)

        monkeypatch.setattr(ctx.jobs, "_create", target)
        create(ctx.jobs)
        assert returned.wait(5)
        try:
            with pytest.raises(RuntimeError, match="still own"):
                ctx.jobs.close(timeout=0.02)
            assert not ctx.made[0].closed and ctx.memory.used == (64 << 10, 1)
        finally:
            release.set()
        ctx.jobs.close(timeout=5)
        assert ctx.made[0].closed and ctx.made[0].starts == 1


def test_failed_owner_does_not_prevent_unrelated_finished_owner_retirement(tmp_path):
    with configured(tmp_path, max_runs=2, blocked=True) as ctx:
        create(ctx.jobs)
        first = ctx.made[0]
        first.failed, first.closing_error = True, True
        first.release.set()
        until(ctx.jobs.failures)
        other = replace(IDENTITY, run_id="other")
        create(ctx.jobs, other)
        second = ctx.made[1]
        second.release.set()
        until(lambda: second.run.ready)
        second.run.done = True
        until(lambda: second.closed)
        until(lambda: ctx.jobs.used == (1, 0))
        assert not first.closed and ctx.memory.used == (64 << 10, 1)


def test_health_failure_retires_without_a_request_for_close(tmp_path):
    with configured(tmp_path) as ctx:
        create(ctx.jobs)
        live(ctx.jobs)
        ctx.made[0].run.controller.failure = "controller failed"
        until(lambda: ctx.jobs.used == (0, 0))
        assert (
            ctx.made[0].interrupts
            and ctx.service.catalog.get(IDENTITY).state == "stopped"
        )


@pytest.mark.parametrize("allocation_stop", [False, True])
def test_actual_child_retirement_preserves_an_existing_controller_borrow(
    tmp_path, monkeypatch, allocation_stop
):
    with live_run(tmp_path, monkeypatch, register=False) as ctx:
        owner = SimpleNamespace(
            run=ctx.live,
            start=ctx.live.start,
            interrupt=ctx.live.interrupt,
            close=ctx.live.close,
        )
        memory = BufferBudget(128 << 20, 1)
        jobs = RunJobs(
            ctx.storage,
            lambda _: owner,
            memory=memory,
            host_bytes=128 << 20,
            max_runs=1,
            reap_interval=0.01,
        )
        jobs.start()
        try:
            jobs.create(
                IDENTITY,
                request_sha256=ctx.configuration.sha256,
                configuration=ctx.configuration,
            )
            live(jobs)
            with jobs.access(IDENTITY) as access:
                controller = access.live()
                if allocation_stop:
                    with pytest.raises(RuntimeError, match="still own"):
                        jobs.close(timeout=0.1)
                    until(lambda: ctx.live.controller.closed)
                    assert not ctx.live.controller.boundary.valid
                else:
                    lease = controller.attach("writer").lease
                    command = OperationRequest(
                        RequestIdentity(
                            "close",
                            "run",
                            1,
                            lease.writer_id,
                            lease.writer_epoch,
                            "server",
                        ),
                        "close",
                    )
                    controller.register(command)
                    controller.receive("close")
                    assert controller.wait("close", timeout=5).state == "completed"
                    until(lambda: next(iter(jobs._jobs.values())).retiring)
                assert jobs.used == (1, 1) and not ctx.live.closed
                assert memory.used == (128 << 20, 1)
                assert ctx.storage._requests.used[1] >= 1
            until(lambda: jobs.used == (0, 0))
            assert ctx.live.closed and ctx.node.closed and memory.used == (0, 0)
            with jobs.access(IDENTITY) as saved:
                assert saved.controller is None
                if not allocation_stop:
                    assert saved.records.get("close")[1].result.operation == "close"
        finally:
            jobs.close(timeout=5)


def test_actual_http_close_retires_automatically_and_keeps_saved_queries(
    tmp_path, monkeypatch
):
    with live_run(tmp_path, monkeypatch, register=False) as ctx:
        owner = SimpleNamespace(
            run=ctx.live,
            start=ctx.live.start,
            interrupt=ctx.live.interrupt,
            close=ctx.live.close,
        )
        memory = BufferBudget(128 << 20, 1)
        jobs = RunJobs(
            ctx.storage,
            lambda _: owner,
            memory=memory,
            host_bytes=128 << 20,
            max_runs=1,
            reap_interval=0.01,
        )
        jobs.start()
        server_memory = BufferBudget(2 * HTTP_SCRATCH_BYTES, 2)
        client_memory = BufferBudget(HTTP_SCRATCH_BYTES, 1)
        token = "private-test-credential-with-more-than-thirty-two-characters"
        http = HTTPService(
            ("127.0.0.1", 0),
            incarnation="server",
            authentication=BearerTokens({token: "owner"}),
            runs=jobs.access,
            artifacts=ctx.storage.artifacts,
            memory=server_memory,
        )
        serving = Thread(target=http.serve_forever, kwargs={"poll_interval": 0.01})
        serving.start()
        client = HTTPTransport(
            f"http://127.0.0.1:{http.server_port}", token=token, memory=client_memory
        )
        try:
            jobs.create(
                IDENTITY,
                request_sha256=ctx.configuration.sha256,
                configuration=ctx.configuration,
            )
            live(jobs)
            route = "/v1/runs/session/run"
            writer = client.request("POST", route + "/writers/writer").value
            command = OperationRequest(
                RequestIdentity(
                    "close", "run", 1, writer.writer_id, writer.writer_epoch, "server"
                ),
                "close",
            )
            client.request("POST", route + "/requests", command)
            client.request("PUT", route + "/requests/close/input")
            result = client.request(
                "GET", route + "/requests/close?wait=completed&timeout=5"
            ).value
            assert result.state == "completed"
            until(lambda: jobs.used == (0, 0))
            assert client.request("POST", route + "/requests", command).value == result
            assert ctx.storage.catalog.get(IDENTITY).state == "closed"
            with ctx.storage.artifacts("owner") as artifacts:
                reference = next(
                    ref for ref in artifacts.page() if ref.id.endswith("stdout")
                )
            output = io.BytesIO()
            client.download("/v1/artifacts/" + reference.id, reference, output)
            assert output.getvalue() == b"service-owned CPU worker exited\n"
        finally:
            http.shutdown()
            http.server_close()
            serving.join(timeout=5)
            assert not serving.is_alive()
            jobs.close(timeout=5)
            assert server_memory.used == client_memory.used == memory.used == (0, 0)
