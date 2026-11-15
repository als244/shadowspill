"""Named Plan handles survive connections and service restart without workers."""

import json
import os
import subprocess
from dataclasses import replace
from urllib.parse import urlencode

import pytest

from engine_tests.server.test_http import OTHER_TOKEN, TOKEN
from engine_tests.server.test_http_creation import endpoint
from engine_tests.server.test_jobs import until
from engine_tests.server.test_local_creation import configured
from engine_tests.server.test_service_client import address
from shadowspill.server.records.artifact_index import ArtifactAlias
from shadowspill_client import ServiceClient
from shadowspill_client.plans import PlansHandle
from shadowspill_client.protocol.plans import SavedPlans
from shadowspill_client.transport.http import RemoteError


@pytest.mark.skipif(
    not os.environ.get("SHADOWSPILL_DEMO_CLIENT_PLANS_SCRIPT"),
    reason="requires the prepared separate NumPy-only demo client",
)
def test_separate_demo_client_saves_and_reuses_plans(tmp_path, monkeypatch):
    with configured(tmp_path, monkeypatch) as ctx, endpoint(ctx) as (server, _):
        result = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                os.environ["SHADOWSPILL_DEMO_CLIENT_HOST"],
                os.environ["SHADOWSPILL_DEMO_CLIENT_PYTHON"],
                "-I",
                os.environ["SHADOWSPILL_DEMO_CLIENT_PLANS_SCRIPT"],
            ],
            input=json.dumps(
                {
                    "ssh_host": "chicago",
                    "port": server._port,
                    "token": TOKEN,
                    "config": ctx.config.to_dict(),
                }
            ),
            text=True,
            capture_output=True,
            timeout=90,
            check=True,
        )
        receipt = json.loads(result.stdout)
        assert all(receipt["absent"].values()) and receipt["client_resources_released"]
        until(lambda: ctx.jobs.used == (0, 0))
        assert len(ctx.owners) == 2 and len(ctx.phases) == 4
        (tmp_path / "demo-plan-names.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )


def test_public_named_plans_survive_close_reconnect_restart_and_skip_preparation(
    tmp_path, monkeypatch
):
    name = "模型 / best + 1"
    with configured(tmp_path, monkeypatch) as ctx, endpoint(ctx) as (server, _):
        with ServiceClient(
            address(server), token=TOKEN, session_id="session"
        ) as client:
            training = client.create_training_client(config=ctx.config).result(10)
            run_id, reference = training.run_id, training.plans.reference
            reader = client.get_training_client(run_id, read_only=True).result(5)
            saved = reader.plans.save(name=name).result(5)
            assert saved.reference == reference
            with ctx.service.artifacts("owner") as store:
                before = store.used
                # The same friendly checkpoint name remains independent.
                store.name(
                    reference, ArtifactAlias("session", run_id, name, reference.id)
                )
            assert training.plans.save(name=name).result(5).reference == reference
            with ctx.service.artifacts("owner") as store:
                assert store.used == before
            training.close().result(5)
            until(lambda: ctx.jobs.used == (0, 0))
            with ctx.service.artifacts("owner") as store:
                before = store.used  # Worker close publishes its logs separately.
            saved.save(name="after close").result(5)
            with ctx.service.artifacts("owner") as store:
                assert store.used == before
                assert store.resolve_alias("session", run_id, name) == reference
        with ServiceClient(
            address(server), token=TOKEN, session_id="session"
        ) as client:
            saved = client.get_plans(run_id, name=name).result(5)
            assert isinstance(saved, PlansHandle) and saved.reference == reference
            assert saved.summary().result(5).plans == reference
            assert len(ctx.owners) == 1 and len(ctx.phases) == 4

    with (
        configured(tmp_path, monkeypatch) as ctx,
        endpoint(ctx) as (server, _),
        ServiceClient(address(server), token=TOKEN, session_id="session") as client,
    ):
        saved = client.get_plans(run_id, name=name).result(5)
        assert saved.reference == reference
        saved.save(name="after restart").result(5)
        assert saved.summary().result(5).plans == reference
        path = saved.download(tmp_path / "saved.ssp").result(5)
        assert path.stat().st_size == reference.size_bytes
        assert not ctx.owners and not ctx.phases and ctx.jobs.used == (0, 0)
        resumed = client.create_training_client(config=ctx.config, plans=saved).result(
            10
        )
        resumed.close().result(5)
        until(lambda: ctx.jobs.used == (0, 0))
        assert len(ctx.owners) == 1 and not ctx.phases


def test_plan_names_reject_wrong_source_and_isolate_authenticated_run_scope(
    tmp_path, monkeypatch
):
    with (
        configured(tmp_path, monkeypatch) as ctx,
        endpoint(ctx) as (server, other),
        ServiceClient(address(server), token=TOKEN, session_id="session") as client,
    ):
        training = client.create_training_client(config=ctx.config).result(10)
        run_id, reference = training.run_id, training.plans.reference
        training.plans.save(name="best").result(5)
        route = f"/v1/runs/session/{run_id}/plans"
        value = SavedPlans("session", run_id, "best", reference)
        for changed in (
            replace(value, session_id="other"),
            replace(value, run_id="other"),
            replace(value, plans=replace(reference, sha256="a" * 64)),
        ):
            with pytest.raises(RemoteError) as invalid:
                server.request("POST", route, changed)
            assert invalid.value.status == 400
        for query in ("", "?name=", "?name=best&name=best", "?other=best"):
            with pytest.raises(RemoteError) as invalid:
                server.request("GET", route + query)
            assert invalid.value.status == 400
        for scoped in (
            route + "?name=missing",
            route.replace("session", "other") + "?name=best",
            route.replace(run_id, "other") + "?name=best",
        ):
            with pytest.raises(RemoteError) as missing:
                server.request("GET", scoped)
            assert missing.value.status == 404
        with pytest.raises(RemoteError) as denied:
            other.request("POST", route, value)
        assert denied.value.status == 404
        with ServiceClient(
            address(server), token=OTHER_TOKEN, session_id="session"
        ) as intruder:
            with pytest.raises(RemoteError) as denied:
                intruder.get_plans(run_id, name="best").result(5)
            assert denied.value.status == 404
        with ctx.service.artifacts("owner") as store:
            assert (
                store.resolve_alias("session", run_id, "best", kind="plans")
                == reference
            )
        training.close().result(5)


def test_client_rejects_mismatched_plan_name_replies_without_retry(
    tmp_path, monkeypatch
):
    with (
        configured(tmp_path, monkeypatch) as ctx,
        endpoint(ctx) as (server, _),
        ServiceClient(address(server), token=TOKEN, session_id="session") as client,
    ):
        training = client.create_training_client(config=ctx.config).result(10)
        training.close().result(5)
        calls = []

        def wrong(method, path, kind, value=None):
            calls.append((method, path))
            return SavedPlans(
                "session", "different-run", "best", training.plans.reference
            )

        monkeypatch.setattr(client, "_request", wrong)
        with pytest.raises(ValueError, match="reply differs"):
            training.plans.save(name="best").result(5)
        with pytest.raises(ValueError, match="another run"):
            client.get_plans(training.run_id, name="best").result(5)
        assert len(calls) == 2
        assert calls[1][1].endswith("?" + urlencode({"name": "best"}))
        with pytest.raises(ValueError, match="requires a training"):
            PlansHandle(training.plans.reference, client).save(name="best")
