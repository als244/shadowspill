"""Client HTML uses durable inspection evidence before training and after close."""

import json
from dataclasses import replace

import pytest

from engine_tests.server.test_http import OTHER_TOKEN, TOKEN
from engine_tests.server.test_http_creation import endpoint
from engine_tests.server.test_jobs import until
from engine_tests.server.test_local_creation import configured
from engine_tests.server.test_service_client import address
from shadowspill.server.artifacts import ArtifactStore
from shadowspill_client import ServiceClient
from shadowspill_client.plans import PlansHandle
from shadowspill_client.protocol.plans import PlanReportRequest
from shadowspill_client.transport.http import RemoteError


def data(path):
    html = path.read_text()
    return json.loads(
        html.split('<script id="data" type="application/json">', 1)[1].split(
            "</script>", 1
        )[0]
    )


def test_report_before_first_step_and_after_close_survives_client_reconnect(
    tmp_path, monkeypatch
):
    with (
        configured(tmp_path, monkeypatch, ranks=2) as ctx,
        endpoint(ctx) as (server, _),
    ):
        with ServiceClient(address(server), token=TOKEN) as service:
            training = service.create_training_client(config=ctx.config).result(10)
            handle = training.plans
            handle.save(name="inspect later").result(5)
            session, run = service.session_id, training.run_id
            original = ArtifactStore._open

            def only_metadata(self, reference):
                assert reference != handle.reference, "report opened executable Plans"
                return original(self, reference)

            monkeypatch.setattr(ArtifactStore, "_open", only_metadata)
            report = handle.report(max_bytes=4 << 20).result(5)
            path = report.download(tmp_path / "before.html").result(5)
            document = data(path)
            rows = document["rows"]
            plans = [row["values"] for row in rows if row["kind"] == "worker_plan"]
            assert {view["rank"] for view in document["views"]} == {0, 1}
            assert len(document["views"]) == sum(len(row["views"]) for row in plans)
            assert {view["id"] for view in document["views"]} == {
                p["id"] for p in plans
            }
            assert len(ctx.phases) == 4
            training.close().result(5)
            assert (
                handle.report(max_bytes=4 << 20).result(5).reference == report.reference
            )
        until(lambda: ctx.http_memory.used == (0, 0))
        with ServiceClient(address(server), token=TOKEN, session_id=session) as service:
            saved = service.get_plans(run, name="inspect later").result(5)
            assert (
                saved.report(max_bytes=4 << 20).result(5).reference == report.reference
            )
            saved.report(max_bytes=4 << 20).result(5).download(
                tmp_path / "after.html"
            ).result(5)
            assert (tmp_path / "after.html").read_bytes() == path.read_bytes()
            wrong = replace(saved, reference=replace(saved.reference, id="different"))
            with pytest.raises(RemoteError) as rejected:
                wrong.report(max_bytes=4 << 20).result(5)
            assert rejected.value.status == 400
        until(lambda: ctx.http_memory.used == (0, 0))
        with ServiceClient(
            address(server), token=OTHER_TOKEN, session_id=session
        ) as other:
            with pytest.raises(RemoteError) as denied:
                PlansHandle(handle.reference, other, run).report(
                    max_bytes=4 << 20
                ).result(5)
            assert denied.value.status == 404


def test_report_output_limit_reclaims_partial_and_does_not_publish_alias(
    tmp_path, monkeypatch
):
    with (
        configured(tmp_path, monkeypatch) as ctx,
        endpoint(ctx) as (server, _),
        ServiceClient(address(server), token=TOKEN) as service,
    ):
        training = service.create_training_client(config=ctx.config).result(10)
        training.close().result(5)
        with ctx.service.artifacts("owner") as store:
            before = store.used
        with pytest.raises(RemoteError):
            training.plans.report(max_bytes=100).result(5)
        until(lambda: ctx.http_memory.used == (0, 0))
        with ctx.service.artifacts("owner") as store:
            assert store.used == before
            assert (
                store.get(
                    PlanReportRequest(
                        service.session_id,
                        training.run_id,
                        training.plans.reference,
                        100,
                    ).report_id
                )
                is None
            )
            assert store._disk.used == (0, 0)
        assert (
            training.plans.report(max_bytes=4 << 20).result(5).reference.size_bytes
            > 100
        )
