"""Public capture delivery over real CPU processes, with sentinel report bytes."""

from dataclasses import replace
from threading import Event

import pytest

from engine_tests.server.test_http import OTHER_TOKEN, TOKEN
from engine_tests.server.test_http_creation import endpoint
from engine_tests.server.test_jobs import until
from engine_tests.server.test_live import CHILD
from engine_tests.server.test_local_creation import configured
from engine_tests.server.test_service_client import address
from engine_tests.training.test_worker_node import collector_workers
from shadowspill.orchestration.capture.files import CollectorLimits
from shadowspill.server import live as live_module
from shadowspill.server.artifacts import ArtifactStore
from shadowspill.server.nsight import publish_manifest
from shadowspill_client import ServiceClient
from shadowspill_client.protocol.captures import NSIGHT_MEDIA_TYPE
from shadowspill_client.transport.http import RemoteError


def settings(config):
    return replace(
        config, execution=replace(config.execution, nsys={"gpu_metrics": False})
    )


def limits(value):
    return replace(
        value,
        nsys=CollectorLimits(
            host_bytes=1 << 20,
            disk_bytes=128 << 10,
            max_files=16,
            finish_seconds=2,
            executable="/bin/true",
        ),
    )


@pytest.mark.parametrize(
    "fault", [None, "missing", "partial", "collector", "publication"]
)
def test_public_capture_pending_final_and_download_survive_close_and_restart(
    tmp_path, monkeypatch, fault
):
    with configured(
        tmp_path, monkeypatch, change=settings, limits_change=limits
    ) as ctx:
        collector_workers(monkeypatch, CHILD, fault=fault)
        original = ArtifactStore.put

        def put(self, ref, source):
            if fault == "publication" and ref.media_type == NSIGHT_MEDIA_TYPE:
                raise OSError("injected optional capture publication failure")
            return original(self, ref, source)

        monkeypatch.setattr(ArtifactStore, "put", put)
        with (
            endpoint(ctx) as (server, _),
            ServiceClient(address(server), token=TOKEN, session_id="session") as client,
        ):
            training = client.create_training_client(config=ctx.config).result(10)
            run_id = training.run_id
            pending = training.nsys().result(5)
            assert pending.record.status == "pending" and pending.report is None
            assert pending.record.plans == training.plans.reference
            reader = client.get_training_client(run_id, read_only=True).result(5)
            assert reader.nsys().result(5).record == pending.record
            node = ctx.owners[0]._node
            startup = node.prepared[0].startup
            assert node.lease.request.host_bytes == (
                ctx.factory.limits.worker.peak_bytes(startup) + (1 << 20)
            )
            assert training.close().result(5).record.state_valid
            until(lambda: ctx.jobs.used == (0, 0))
            final = training.nsys().result(5)
            status = (
                "ready"
                if fault is None
                else "missing"
                if fault == "missing"
                else "failed"
            )
            assert final.record.status == status
            assert (final.report is not None) == (
                fault not in ("missing", "publication")
            )
            if final.report is not None:
                path = final.report.download(tmp_path / "capture.nsys-rep").result(5)
                assert path.read_bytes() == b"CPU sentinel report"
            with ctx.service.artifacts("owner") as store:
                before = store.used
                assert training.nsys().result(5).record == final.record
                assert store.used == before  # Read-only lookup never rewrites journals.
            assert node.closed and node.lease.closed
            assert ctx.disk.used == (0, 0) and not ctx.node.used.devices
    with (
        configured(tmp_path, monkeypatch) as ctx,
        endpoint(ctx) as (server, _),
        ServiceClient(address(server), token=TOKEN, session_id="session") as client,
    ):
        reopened = client.get_nsys(run_id).result(5)
        assert reopened.record == final.record
        if reopened.report is not None:
            path = reopened.report.download(tmp_path / "reopened.nsys-rep").result(5)
            assert path.read_bytes() == b"CPU sentinel report"
        assert not ctx.owners and not ctx.phases and not ctx.node.used.devices


def test_capture_routes_are_owner_scoped_and_validate_rank_and_response(
    tmp_path, monkeypatch
):
    with configured(
        tmp_path, monkeypatch, change=settings, limits_change=limits
    ) as ctx:
        collector_workers(monkeypatch, CHILD)
        with (
            endpoint(ctx) as (server, other),
            ServiceClient(address(server), token=TOKEN, session_id="session") as client,
        ):
            training = client.create_training_client(config=ctx.config).result(10)
            pending = training.nsys().result(5).record
            route = f"/v1/runs/session/{training.run_id}/nsys"
            for query in (
                "",
                "?rank=",
                "?rank=0&rank=0",
                "?rank=-1",
                "?rank=65536",
                "?other=0",
            ):
                with pytest.raises(RemoteError) as invalid:
                    server.request("GET", route + query)
                assert invalid.value.status == 400
            for rank in (-1, True, 65536):
                with pytest.raises((ValueError, TypeError)):
                    training.nsys(rank=rank)
            for scoped in (
                route + "?rank=1",
                route.replace("session", "other") + "?rank=0",
            ):
                with pytest.raises(RemoteError) as missing:
                    server.request("GET", scoped)
                assert missing.value.status == 404
            with pytest.raises(RemoteError) as denied:
                other.request("GET", route + "?rank=0")
            assert denied.value.status == 404
            with ServiceClient(
                address(server), token=OTHER_TOKEN, session_id="session"
            ) as intruder:
                with pytest.raises(RemoteError) as denied:
                    intruder.get_nsys(training.run_id).result(5)
                assert denied.value.status == 404
            training.close().result(5)
            until(lambda: ctx.jobs.used == (0, 0))
            calls = []

            def wrong(method, path, kind, value=None):
                calls.append((method, path))
                return replace(pending, rank=1)

            monkeypatch.setattr(client, "_request", wrong)
            with pytest.raises(ValueError, match="another run or rank"):
                training.nsys().result(5)
            assert len(calls) == 1


def test_unfinalized_capture_is_missing_after_terminal_run_recovery(
    tmp_path, monkeypatch
):
    with configured(
        tmp_path, monkeypatch, change=settings, limits_change=limits
    ) as ctx:
        collector_workers(monkeypatch, CHILD)
        with (
            endpoint(ctx) as (server, _),
            ServiceClient(address(server), token=TOKEN, session_id="session") as client,
        ):
            training = client.create_training_client(config=ctx.config).result(10)
            run_id = training.run_id
            pending = training.nsys().result(5).record
            training.close().result(5)
            until(lambda: ctx.jobs.used == (0, 0))
            # Simulate the recovered journal containing only pre-launch evidence.
            with ctx.service.artifacts("owner") as store:
                publish_manifest(store, pending)
    with (
        configured(tmp_path, monkeypatch) as ctx,
        endpoint(ctx) as (server, _),
        ServiceClient(address(server), token=TOKEN, session_id="session") as client,
    ):
        capture = client.get_nsys(run_id).result(5)
        assert capture.record.status == capture.record.window.state == "missing"
        assert "without a finalized" in capture.record.error
        assert capture.report is None and not ctx.owners


def test_missing_installed_collector_rejects_before_preparation(tmp_path, monkeypatch):
    from engine_tests.server.test_local_creation import submit
    from engine_tests.server.test_storage import IDENTITY

    def absent(value):
        return replace(
            limits(value),
            nsys=replace(limits(value).nsys, executable="shadowspill-absent-nsys"),
        )

    with configured(
        tmp_path, monkeypatch, change=settings, limits_change=absent
    ) as ctx:
        submit(ctx)
        until(lambda: ctx.service.catalog.get(IDENTITY).state == "failed")
        until(lambda: ctx.jobs.used == (0, 0))
        assert not ctx.phases and not ctx.node.used.devices
        assert "executable" in ctx.service.catalog.get(IDENTITY).error.message


def test_manifest_failure_retains_files_and_retries_after_confirmed_worker_exit(
    tmp_path, monkeypatch
):
    failed, retry = Event(), Event()
    finalize = live_module.finalize_manifest

    def publish(*args):
        if not retry.is_set():
            failed.set()
            raise OSError("manifest storage unavailable")
        return finalize(*args)

    with configured(
        tmp_path, monkeypatch, change=settings, limits_change=limits
    ) as ctx:
        collector_workers(monkeypatch, CHILD)
        with (
            endpoint(ctx) as (server, _),
            ServiceClient(address(server), token=TOKEN, session_id="session") as client,
        ):
            training = client.create_training_client(config=ctx.config).result(10)
            owner = ctx.owners[0]
            node = owner._node
            monkeypatch.setattr(live_module, "finalize_manifest", publish)
            try:
                result = training.close().result(5)
                assert result.state_valid and failed.wait(5)
                assert node.lease.closed and not node.closed and not owner.closed
                assert ctx.disk.used[0] > 0 and not ctx.node.used.devices
                assert training.nsys().result(5).record.status == "pending"
                assert all(
                    capture.directory.exists() for capture in node.captures.values()
                )
            finally:
                retry.set()
            until(lambda: ctx.jobs.used == (0, 0))
            capture = training.nsys().result(5)
            assert capture.record.status == "ready"
            assert node.closed and owner.closed and ctx.disk.used == (0, 0)
            assert (
                training.requests.get(result.identity.request_id).result(5).record
                == result.record
            )
