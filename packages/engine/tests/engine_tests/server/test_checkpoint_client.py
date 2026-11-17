"""Public saves/downloads use actual HTTP, journal and owned CPU rank encoders."""

import hashlib
from concurrent.futures import TimeoutError
from contextlib import contextmanager
from threading import Event

import pytest

from engine_tests.server.test_http import TOKEN, controller_lifecycle, endpoint
from engine_tests.server.test_request_records import IDENTITY
from engine_tests.server.test_request_records import store as records_store
from engine_tests.server.test_service_client import address
from engine_tests.training.test_prepared_requests import family
from engine_tests.training.test_worker_checkpoints import running
from shadowspill.server.http import RunAccess
from shadowspill.server.startup.checkpoint import CheckpointResolver
from shadowspill.training.session.controller import RunController
from shadowspill.training.session.prepared import PreparedRequests
from shadowspill.training.session.state import OperationState
from shadowspill_client import ServiceClient, TrainingClient
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.checkpoints import CheckpointHandle
from shadowspill_client.protocol.codec import decode
from shadowspill_client.protocol.runs import RunInfo


@contextmanager
def connected(root, *, completed_updates=0, weights=False):
    with (
        endpoint(root) as http,
        records_store(root) as records,
        running(root, http.artifacts, count=1, weights=weights) as (
            workers,
            publisher,
            _,
        ),
    ):
        plans = family()
        prepared = PreparedRequests(
            plans,
            cache_root=root,
            checkpoint=CheckpointResolver(
                http.artifacts,
                session_id="session",
                world_size=1,
                configuration=plans.inventory.layouts.setup.config,
            ),
        )
        config = prepared.setup.config
        inputs = BufferBudget(
            config.execution.max_staging_bytes, config.execution.max_pending_requests
        )
        controller = RunController(
            records,
            prepared,
            workers,
            publisher,
            server_incarnation="server",
            budget=inputs,
            state=OperationState(
                history_id="history", completed_updates=completed_updates
            ),
        )

        @contextmanager
        def runs(identity):
            assert identity == IDENTITY
            yield RunAccess(records, controller)

        http.service.runs = runs
        try:
            with ServiceClient(
                address(http.client), token=TOKEN, session_id="session"
            ) as service:
                info = RunInfo(
                    "owner",
                    "session",
                    "run",
                    "server",
                    "a" * 64,
                    http.artifacts.get("config"),
                    state="ready",
                    stage=None,
                    plans=http.artifacts.get("plans"),
                    initial_history_id="history",
                )
                http.service.jobs = controller_lifecycle(info, lambda: workers.closed)
                client = TrainingClient(service, info, config, records.writer)
                yield client, service, http, records, controller, workers
        finally:
            controller.stop(workers.close, timeout=5)
        assert inputs.used == (0, 0)


def test_public_save_freezes_metadata_orders_close_and_downloads_after_worker_exit(
    tmp_path, monkeypatch
):
    with connected(tmp_path) as (training, service, http, records, controller, workers):
        entered, release = Event(), Event()
        original = http.artifacts._commit

        def delayed(*args):
            entered.set()
            assert release.wait(5)
            return original(*args)

        monkeypatch.setattr(http.artifacts, "_commit", delayed)
        metadata = {"cursor": [1, 4]}
        pending = training.save_checkpoint(name="latest", metadata=metadata)
        metadata["cursor"].append(9)
        try:
            assert pending.accepted().result(10).identity == pending.identity
            assert entered.wait(10)
            later = training.close()
            with pytest.raises(TimeoutError):
                pending.result(0)
            assert not pending.cancel() and not later.done()
            assert (tmp_path / "executions-0").read_text() == "1\n"
        finally:
            release.set()
        saved = pending.result(10)
        assert isinstance(saved, CheckpointHandle)
        assert saved.id == http.artifacts.resolve_alias("session", "run", "latest").id
        assert later.result(10).completed_updates == 0 and workers.closed
        assert records.get(later.request_id)[0].dependencies == (pending.request_id,)
        assert (tmp_path / "executions-0").read_text() == "1\n2\n"
        looked_up = training.requests.get(pending.request_id).result(5)
        assert (
            looked_up.checkpoint == saved
            and looked_up.record.checkpoint == saved.reference
        )
        assert service.artifact(saved.reference) == saved
        manifest = saved.manifest().result(5)
        assert manifest.info.metadata["cursor"] == (1, 4)
        assert manifest.info.request_id == pending.request_id
        destination = saved.download(tmp_path / "copy").result(10)
        assert decode((destination / "manifest.json").read_bytes()) == manifest
        for row in manifest.ranks:
            with (destination / "ranks" / f"rank-{row.rank:05d}.ssp").open(
                "rb"
            ) as source:
                assert (
                    hashlib.file_digest(source, "sha256").hexdigest()
                    == row.archive.sha256
                )
        with pytest.raises(FileExistsError):
            saved.download(destination).result(5)
        assert (destination / "manifest.json").exists()
        assert controller.boundary.valid and controller.boundary.closed


def test_failed_rank_download_discards_own_partial_copy_and_preserves_saved_result(
    tmp_path, monkeypatch
):
    with connected(tmp_path) as (training, service, _, records, controller, _):
        pending = training.save_checkpoint()
        saved = pending.result(10)
        manifest = saved.manifest().result(5)
        original = service._transport.download

        def fail(route, reference, destination):
            if reference.id == manifest.ranks[0].archive.id:
                destination.write(b"partial")
                raise OSError("download interrupted")
            return original(route, reference, destination)

        destination = tmp_path / "failed-copy"
        unrelated = tmp_path / "keep"
        unrelated.write_text("unrelated")
        with monkeypatch.context() as patch:
            patch.setattr(service._transport, "download", fail)
            with pytest.raises(OSError, match="interrupted"):
                saved.download(destination).result(5)
        assert not destination.exists() and unrelated.read_text() == "unrelated"
        assert records.get(pending.request_id)[1].state == "completed"
        assert controller.boundary.valid
        assert saved.download(destination).result(5) == destination
        assert training.close().result(10).completed_updates == 0
