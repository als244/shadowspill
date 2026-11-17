"""Actual HTTP/job/process ownership for measurement-only preparation."""

import json
import sys
import zipfile
from dataclasses import replace
from threading import Event

import pytest

from engine_tests.server.test_http import TOKEN
from engine_tests.server.test_http_creation import endpoint
from engine_tests.server.test_jobs import until
from engine_tests.server.test_local_creation import configured
from engine_tests.server.test_service_client import address
from engine_tests.server.test_storage import storage
from engine_tests.training.test_setup_resolution import installed
from shadowspill.orchestration.launch.phases import PhaseInvocation
from shadowspill.server.records.requests import RunIdentity
from shadowspill.server.startup import creation as module
from shadowspill.server.startup import initialization
from shadowspill.training.preparation.phases import job
from shadowspill_client import ServiceClient
from shadowspill_client._async import ConnectionClosed
from shadowspill_client.creation import CreationError
from shadowspill_client.protocol.creation import TrainingCreation
from shadowspill_client.protocol.requests import ArtifactRef


def measurement_phases(monkeypatch, *, fail=False):
    phases = []
    monkeypatch.setattr(job, "load_registry", lambda _: installed())

    def measure(root, *, packages, config, environment, launch):
        assert config.model.initialize is None
        if config.preparation.reuse:
            assert (root / "reused.json").is_file()
            assert job.PreparationJob.load(root / job.JOB_NAME).config == config
        else:
            root.mkdir()
            job.PreparationJob(tuple(packages), config).save(root / job.JOB_NAME)
        for phase in ("build", "profile", "communicate"):
            phases.append(phase)
            result = launch(
                PhaseInvocation(
                    phase,
                    (sys.executable, "-c", f"print({phase!r})"),
                    environment,
                    1,
                    root,
                    root / (phase + ".log"),
                )
            )
            assert result.returncode == 0
            if fail and phase == "profile":
                raise RuntimeError("intentional measurement failure")
        _, _, inventory = job.load_job(root)
        for stage in ("programs", "profiled", "calibrated", "communication"):
            (root / stage).mkdir(exist_ok=True)
            for key in inventory.templates:
                (root / stage / f"{key}.ssp").write_bytes(f"{stage}:{key}".encode())

    def forbidden(*args, **kwargs):
        raise AssertionError("measurement-only job attempted planning/initialization")

    monkeypatch.setattr(module, "measure_job", measure)
    monkeypatch.setattr(initialization, "prepare_initialization", forbidden)
    for name in ("prepare_job", "WorkerNode"):
        monkeypatch.setattr(module, name, forbidden)
    return phases


def no_initializer(config):
    return replace(config, model=replace(config.model, initialize=None))


@pytest.mark.parametrize("disconnect", [False, True])
def test_public_measurement_waits_for_retirement_and_rejoins_without_replay(
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

    with (
        configured(tmp_path, monkeypatch, change=no_initializer, fault=fault) as ctx,
        endpoint(ctx) as (server, _),
        ServiceClient(address(server), token=TOKEN, session_id="preparation") as client,
    ):
        phases = measurement_phases(monkeypatch)
        future = client.prepare_training(config=ctx.config)
        identity = RunIdentity("owner", future.session_id, future.run_id)
        route = f"/v1/runs/{future.session_id}/{future.run_id}"
        try:
            assert closed.wait(10), ctx.jobs.failures()
            reply = server.request("GET", route)
            assert reply.status == 202 and reply.value.state == "prepared"
            assert not future.cancel()
            with pytest.raises(TimeoutError):
                future.result(0.02)
            assert ctx.owners[0].run is None
            assert not ctx.service._request_path(identity).exists()
            with (
                pytest.raises(ValueError, match="no training request journal"),
                ctx.service.requests(identity),
            ):
                pass
            if disconnect:
                client.close()
                with pytest.raises(ConnectionClosed):
                    future.result(5)
        finally:
            release.set()
        until(lambda: ctx.jobs.used == (0, 0))
        with ServiceClient(
            address(server), token=TOKEN, session_id=future.session_id
        ) as rejoined:
            prepared = rejoined.get_preparation(future.run_id).result(5)
            if not disconnect:
                assert future.result(5).info == prepared.info
            artifact = prepared.export().result(5)
            destination = artifact.download(tmp_path / "measured.zip").result(5)
            with zipfile.ZipFile(destination) as archive:
                manifest = json.loads(archive.read("manifest.json"))
                assert manifest["captures"] and manifest["files"]
                assert manifest["configuration"]["model"]["initialize"] is None
            assert (
                server.request(
                    "POST", route, TrainingCreation(ctx.config, kind="preparation")
                ).value
                == prepared.info
            )
            with pytest.raises(ValueError, match="no training client"):
                rejoined.get_training_client(future.run_id).result(5)
        assert phases == ["build", "profile", "communicate"]
        assert len(ctx.owners) == 1 and ctx.owners[0].closed
        assert not ctx.jobs.failures() and ctx.disk.used == ctx.job_memory.used == (
            0,
            0,
        )


def test_failed_measurement_is_visible_and_never_replays(tmp_path, monkeypatch):
    with (
        configured(tmp_path, monkeypatch, change=no_initializer) as ctx,
        endpoint(ctx) as (server, _),
        ServiceClient(address(server), token=TOKEN) as client,
    ):
        phases = measurement_phases(monkeypatch, fail=True)
        future = client.prepare_training(config=ctx.config)
        with pytest.raises(CreationError, match="intentional measurement failure"):
            future.result(10)
        with pytest.raises(CreationError, match="intentional measurement failure"):
            client.get_preparation(future.run_id).result(10)
        until(lambda: ctx.jobs.used == (0, 0))
        assert phases == ["build", "profile"] and len(ctx.owners) == 1
        assert ctx.owners[0].run is None and not ctx.jobs.failures()


def test_restart_stops_measurement_without_creating_training_history(tmp_path):
    identity = RunIdentity("owner", "session", "measurement")
    with storage(tmp_path, incarnation="old") as service:
        service.catalog.create(
            identity,
            request_sha256="a" * 64,
            configuration=ArtifactRef("config", "b" * 64, 100),
            kind="preparation",
        )
        service.catalog.progress(identity, "profile")
    with storage(tmp_path, incarnation="replacement") as service:
        service.recover()
        stopped = service.catalog.get(identity)
        assert stopped.state == "stopped" and stopped.kind == "preparation"
        assert "new measurement job" in stopped.error.message
        assert stopped.preparation_export is None
        assert not service._request_path(identity).exists()


def test_public_reuse_authenticates_the_original_measurement_job(tmp_path, monkeypatch):
    with (
        configured(
            tmp_path, monkeypatch, change=no_initializer, artifact_count=300
        ) as ctx,
        endpoint(ctx) as (server, _),
        ServiceClient(address(server), token=TOKEN, session_id="source") as client,
    ):
        phases = measurement_phases(monkeypatch)
        original = client.prepare_training(config=ctx.config).result(10)
        config = replace(
            ctx.config,
            preparation=replace(
                ctx.config.preparation,
                reuse=original.reuse,
            ),
        )
        reused = client.prepare_training(config=config).result(10)
        archive = reused.export().result(5).download(tmp_path / "reused.zip").result(5)
        with zipfile.ZipFile(archive) as exported:
            receipt = json.loads(exported.read("reused.json"))
            assert receipt["source"] == original.info.preparation_export.to_dict()
            assert receipt["calibration_reused"]
        assert phases == ["build", "profile", "communicate"] * 2
        assert ctx.jobs.used == (0, 0)

        # Another owner's real completed record is not a reuse grant.
        foreign = RunIdentity("someone_else", "foreign", original.info.run_id)
        ctx.service.catalog.create(
            foreign,
            request_sha256="c" * 64,
            configuration=original.info.configuration,
            kind="preparation",
        )
        ctx.service.catalog.progress(foreign, "communicate")
        ctx.service.catalog.prepared(foreign, original.info.preparation_export)
        missing = replace(
            config,
            preparation=replace(
                config.preparation,
                reuse={
                    "measurements": {
                        "session_id": "foreign",
                        "run_id": original.info.run_id,
                    },
                },
            ),
        )
        with pytest.raises(CreationError, match="not a completed preparation owned"):
            client.prepare_training(config=missing).result(10)
        assert len(phases) == 6
        until(lambda: ctx.jobs.used == (0, 0))
