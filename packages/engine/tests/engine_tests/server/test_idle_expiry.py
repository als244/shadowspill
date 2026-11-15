"""Opt-in expiry releases actual CPU sentinel workers and retains catalog evidence."""

from dataclasses import replace

from engine_tests.server.test_http import TOKEN
from engine_tests.server.test_http_creation import endpoint
from engine_tests.server.test_jobs import until
from engine_tests.server.test_local_creation import configured
from engine_tests.server.test_service_client import address
from shadowspill.server.records.requests import RunIdentity
from shadowspill_client import ServiceClient
from shadowspill_client.protocol.runs import RunInfo


def test_public_idle_expiry_releases_workers_without_polling_and_survives_restart(
    tmp_path, monkeypatch
):
    def change(config):
        return replace(
            config, execution=replace(config.execution, idle_timeout_seconds=0.5)
        )

    with (
        configured(tmp_path, monkeypatch, change=change) as ctx,
        endpoint(ctx) as (server, _),
        ServiceClient(address(server), token=TOKEN, session_id="session") as client,
    ):
        training = client.create_training_client(config=ctx.config).result(10)
        identity = RunIdentity("owner", "session", training.run_id)
        until(lambda: ctx.jobs.used == (0, 0))
        info = ctx.service.catalog.get(identity)
        assert info.state == "stopped" and info.error.code == "idle_timeout"
        assert ctx.owners[0].closed and ctx.owners[0].run.closed
        assert not ctx.node.used.devices and ctx.disk.used == (0, 0)
        assert training.plans.summary().result(5).plans == info.plans
        saved = training.plans.save(name="expired-run").result(5)
        assert saved.reference == info.plans
    with (
        configured(tmp_path, monkeypatch) as ctx,
        endpoint(ctx) as (server, _),
        ServiceClient(address(server), token=TOKEN, session_id="session") as client,
    ):
        restored = client._request(
            "GET", f"/v1/runs/session/{identity.run_id}", RunInfo
        )
        assert restored == info
        assert (
            client.get_plans(identity.run_id, name="expired-run").result(5).reference
            == info.plans
        )
        assert not ctx.owners and not ctx.phases and ctx.jobs.used == (0, 0)
