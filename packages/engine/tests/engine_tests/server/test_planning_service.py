"""Public planning publishes reports and releases resources before initialization."""

from dataclasses import replace
from threading import Event

import pytest

from engine_tests.server.test_http import TOKEN
from engine_tests.server.test_http_creation import endpoint
from engine_tests.server.test_jobs import until
from engine_tests.server.test_local_creation import configured
from engine_tests.server.test_plan_reports import data
from engine_tests.server.test_preparation_service import no_initializer
from engine_tests.server.test_service_client import address
from engine_tests.server.test_storage import storage
from shadowspill.server.records.requests import RunIdentity
from shadowspill.server.startup import creation as module
from shadowspill.server.startup import initialization
from shadowspill_client import ServiceClient
from shadowspill_client._async import ConnectionClosed
from shadowspill_client.creation import CreationError
from shadowspill_client.protocol.requests import ArtifactRef


@pytest.mark.parametrize("disconnect", [False, True])
def test_plans_and_reports_wait_for_cleanup_without_training_state(
    tmp_path, monkeypatch, disconnect
):
    closed, release = Event(), Event()

    def fault(owner):
        original = owner.close

        def close():
            original()
            closed.set()
            assert release.wait(10)

        monkeypatch.setattr(owner, "close", close)

    def forbidden(*args, **kwargs):
        raise AssertionError("planning attempted training initialization")

    with (
        configured(tmp_path, monkeypatch, change=no_initializer, fault=fault) as ctx,
        endpoint(ctx) as (server, _),
        ServiceClient(address(server), token=TOKEN) as service,
    ):
        monkeypatch.setattr(initialization, "prepare_initialization", forbidden)
        monkeypatch.setattr(module, "WorkerNode", forbidden)
        future = service.plan_training(config=ctx.config)
        identity = RunIdentity("owner", future.session_id, future.run_id)
        try:
            assert closed.wait(10), ctx.jobs.failures()
            reply = server.request(
                "GET", f"/v1/runs/{future.session_id}/{future.run_id}"
            )
            assert reply.status == 202 and reply.value.state == "planned"
            assert reply.value.initial_history_id is None
            assert ctx.owners[0].run is None
            assert not ctx.service._request_path(identity).exists()
            with pytest.raises(TimeoutError):
                future.result(0.02)
            if disconnect:
                service.close()
                with pytest.raises(ConnectionClosed):
                    future.result(5)
        finally:
            release.set()
        until(lambda: ctx.jobs.used == (0, 0))
        with ServiceClient(
            address(server), token=TOKEN, session_id=future.session_id
        ) as rejoined:
            plans = rejoined.get_planning(future.run_id).result(5)
            report = plans.report(max_bytes=4 << 20).result(5)
            path = report.download(tmp_path / "plan.html").result(5)
            assert data(path)["views"]
            plans.save(name="before training").result(5)
            assert (
                rejoined.get_plans(future.run_id, name="before training")
                .result(5)
                .reference
                == plans.reference
            )
            if not disconnect:
                assert future.result(5).reference == plans.reference
            with pytest.raises(ValueError, match="no training client"):
                rejoined.get_training_client(future.run_id).result(5)
            with pytest.raises(ValueError, match="not a preparation job"):
                rejoined.get_preparation(future.run_id).result(5)
        assert ctx.phases == ["build", "profile", "communicate", "plan"]
        assert len(ctx.owners) == 1 and not ctx.jobs.failures()


def test_training_from_planned_artifact_skips_all_preparation(tmp_path, monkeypatch):
    with (
        configured(tmp_path, monkeypatch) as ctx,
        endpoint(ctx) as (server, _),
        ServiceClient(address(server), token=TOKEN) as service,
    ):
        plans = service.plan_training(config=ctx.config).result(10)
        assert ctx.jobs.used == (0, 0) and ctx.owners[0].run is None

        def forbidden(*args, **kwargs):
            raise AssertionError("saved Plans rebuilt")

        monkeypatch.setattr(module, "prepare_job", forbidden)
        training = service.create_training_client(
            config=ctx.config, plans=plans
        ).result(10)
        assert training.plans.reference == plans.reference
        training.close().result(5)
        until(lambda: ctx.jobs.used == (0, 0))
        assert ctx.phases == ["build", "profile", "communicate", "plan"]
        assert len(ctx.owners) == 2 and not ctx.jobs.failures()


def test_failed_planning_can_be_inspected_without_replay(tmp_path, monkeypatch):
    with (
        configured(tmp_path, monkeypatch) as ctx,
        endpoint(ctx) as (server, _),
        ServiceClient(address(server), token=TOKEN) as service,
    ):
        calls = []

        def fail(*args, **kwargs):
            calls.append(1)
            raise RuntimeError("intentional planning failure")

        monkeypatch.setattr(module, "prepare_job", fail)
        future = service.plan_training(config=ctx.config)
        with pytest.raises(CreationError, match="intentional planning failure"):
            future.result(10)
        with pytest.raises(CreationError, match="intentional planning failure"):
            service.get_planning(future.run_id).result(10)
        until(lambda: ctx.jobs.used == (0, 0))
        assert calls == [1] and ctx.owners[0].run is None


def test_planning_recovery_keeps_completed_plans_and_stops_unfinished_job(tmp_path):
    finished = RunIdentity("owner", "session", "finished")
    interrupted = replace(finished, run_id="interrupted")
    plans = ArtifactRef("plans", "c" * 64, 100)
    with storage(tmp_path, incarnation="old") as service:
        for identity in (finished, interrupted):
            service.catalog.create(
                identity,
                request_sha256="a" * 64,
                configuration=ArtifactRef("config", "b" * 64, 100),
                kind="planning",
            )
            service.catalog.progress(identity, "plan")
        service.catalog.planned(finished, plans)
    with storage(tmp_path, incarnation="replacement") as service:
        service.recover()
        assert service.catalog.get(finished).plans == plans
        assert service.catalog.get(finished).state == "planned"
        assert service.catalog.get(interrupted).state == "stopped"
        assert not service._request_path(interrupted).exists()
