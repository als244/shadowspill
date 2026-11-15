"""Public inference export uses actual CPU workers, HTTP and durable publication."""

import hashlib
from contextlib import contextmanager
from dataclasses import replace
from threading import Event

import pytest
import torch

from engine_tests.checkpoint.test_conversion import WEIGHT
from engine_tests.checkpoint.test_exports import decode as decode_rank
from engine_tests.server.test_artifacts import store
from engine_tests.server.test_checkpoint_client import connected
from engine_tests.server.test_http import TOKEN, endpoint
from engine_tests.server.test_request_records import store as records_store
from engine_tests.server.test_service_client import address
from engine_tests.training.test_worker_checkpoints import dispatch, running, save_call
from shadowspill.training.session.dispatch import RequestDispatcher
from shadowspill.training.session.state import OperationState
from shadowspill_client import ServiceClient
from shadowspill_client.checkpoints import CheckpointHandle
from shadowspill_client.protocol.checkpoints import WeightsManifest
from shadowspill_client.protocol.codec import ProtocolError, decode
from shadowspill_client.transport.http import TransportError
from shadowspill_client.weights import WeightsHandle


@pytest.mark.parametrize("weights", ["master", "compute"])
def test_public_export_orders_close_and_copies_to_fresh_service(
    tmp_path, monkeypatch, weights
):
    with connected(tmp_path, weights=True, completed_updates=7) as (
        training,
        service,
        http,
        records,
        controller,
        workers,
    ):
        entered, release = Event(), Event()
        original = http.artifacts._commit

        def delayed(*args):
            entered.set()
            assert release.wait(5)
            return original(*args)

        monkeypatch.setattr(http.artifacts, "_commit", delayed)
        metadata = {"cursor": [3, 7]}
        pending = training.export_weights(
            name="adapters", metadata=metadata, weights=weights
        )
        metadata["cursor"].append(9)
        try:
            assert pending.accepted().result(10).identity == pending.identity
            assert entered.wait(10)
            later = training.close()
            assert not pending.done() and not later.done()
        finally:
            release.set()
        saved = pending.result(10)
        assert isinstance(saved, WeightsHandle) and not isinstance(
            saved, CheckpointHandle
        )
        assert service.artifact(saved.reference) == saved
        assert later.result(10).completed_updates == 7 and workers.closed
        result = training.requests.get(pending.request_id).result(5)
        assert result.weights == saved and result.checkpoint is None
        assert records.get(later.request_id)[0].dependencies == (pending.request_id,)
        manifest = saved.manifest().result(5)
        assert isinstance(manifest, WeightsManifest)
        assert manifest.info.completed_updates == 7
        assert manifest.info.metadata["cursor"] == (3, 7)
        destination = saved.download(tmp_path / "weights", resume=True).result(10)
        assert decode((destination / "manifest.json").read_bytes()) == manifest
        raw = (destination / "ranks/rank-00000.ssp").read_bytes()
        assert hashlib.sha256(raw).hexdigest() == manifest.ranks[0].archive.sha256
        rank, tensors = decode_rank(raw)
        expected = "master.linear.weight" if weights == "master" else "linear.weight"
        assert set(tensors) == {expected} and rank["format"] == "shadowspill.weights"
        torch.testing.assert_close(
            tensors[expected], WEIGHT.flatten().to(tensors[expected].dtype)
        )
        assert controller.boundary.valid
    other = tmp_path / "other"
    other.mkdir()
    with (
        endpoint(other) as http,
        ServiceClient(address(http.client), token=TOKEN) as service,
    ):
        uploaded = service.artifacts.upload(
            destination, reference=saved.reference
        ).result(10)
        assert isinstance(uploaded, WeightsHandle)
        assert uploaded.reference == saved.reference
        assert uploaded.manifest().result(5) == manifest
        copy = uploaded.download(tmp_path / "downloaded").result(10)
        assert (copy / "ranks/rank-00000.ssp").read_bytes() == raw


def test_export_rejects_open_accumulation_unknown_format_and_optimizer_restore(
    tmp_path,
):
    with connected(tmp_path, weights=True) as (training, _, _, _, controller, workers):
        # The shared state machine refuses the export before any worker is released.
        state = controller._dispatcher._state
        state.complete(
            state.begin("forward_backward", "contribution", participates=True)
        )
        with pytest.raises(Exception, match="finish optimization"):
            training.export_weights().result(10)
        assert not (tmp_path / "executions-0").exists()
        training.discard_gradients().result(10)
        with pytest.raises(Exception, match="no installed weights exporter"):
            training.export_weights(format="uninstalled").result(10)
        assert (tmp_path / "executions-0").read_text() == "1\n"
        saved = training.export_weights(name="weights").result(10)
        with pytest.raises(Exception, match="checkpoint"):
            training.load_checkpoint(saved.id).result(10)
        assert controller.boundary.valid and not workers.closed
        training.close().result(10)


@pytest.mark.parametrize(
    "failure",
    [None, "capacity", "no_encoder", "lost_completion", "receipt", "publication"],
)
def test_failed_inference_generation_never_publishes_a_partial_rank_set(
    tmp_path, monkeypatch, failure
):
    with (
        store(tmp_path) as artifacts,
        records_store(tmp_path) as records,
        running(tmp_path, artifacts, weights=True, mode=failure or "normal") as (
            workers,
            publisher,
            _,
        ),
    ):
        before = artifacts.used
        call = save_call()
        call = replace(
            call,
            request=replace(
                call.request, save=replace(call.request.save, kind="weights")
            ),
        )
        state = OperationState(completed_updates=9)
        dispatcher = RequestDispatcher(records, state, max_pending=4)
        if failure == "capacity":
            artifacts.max_bytes = artifacts.used[0] + 1
        elif failure == "receipt":

            def fail(_):
                raise OSError("receipt failed")

            monkeypatch.setattr(records, "executed", fail)
        elif failure == "publication":

            def fail(*_):
                raise OSError("publication failed")

            monkeypatch.setattr(artifacts, "_commit", fail)
        if failure == "receipt":
            with pytest.raises(OSError, match="receipt failed"):
                dispatch(records, dispatcher, workers, publisher, call)
        else:
            result = dispatch(records, dispatcher, workers, publisher, call)
            assert result.state == ("completed" if failure is None else "failed")
        if failure is None:
            assert state.boundary.valid and state.boundary.completed_updates == 9
            reference = result.result.weights
            assert artifacts.resolve_alias("session", "run", "latest") == reference
            with artifacts.open(reference) as stream:
                manifest = decode(stream.read())
            assert isinstance(manifest, WeightsManifest) and len(manifest.ranks) == 2
            for row in manifest.ranks:
                with artifacts.open(row.archive) as stream:
                    rank, tensors = decode_rank(stream.read())
                start = (1 - row.rank) * 2
                assert rank["metadata"]["weights"]["completed_updates"] == 9
                assert set(tensors) == {"master.linear.weight"}
                torch.testing.assert_close(
                    tensors["master.linear.weight"], WEIGHT[start : start + 2].flatten()
                )
            return
        assert artifacts.resolve_alias("session", "run", "latest") is None
        assert artifacts.used == before
        if failure in ("capacity", "no_encoder"):
            assert state.boundary.valid and not list(tmp_path.glob("executions-*"))
        elif failure == "publication":
            assert (
                state.boundary.valid and result.failure.execution_effect == "completed"
            )
        else:
            assert not state.boundary.valid


def test_interrupted_weights_download_resumes_after_worker_exit(tmp_path, monkeypatch):
    with connected(tmp_path, weights=True) as (training, _, http, *_):
        saved = training.export_weights().result(10)
        training.close().result(10)
        rank = saved.manifest().result(5).ranks[0].archive
        original, observations = http.artifacts.open, []

        @contextmanager
        def opened(reference):
            with original(reference) as stream:
                if reference != rank:
                    yield stream
                    return
                offsets = []
                observations.append(offsets)

                class Source:
                    def seek(self, offset):
                        offsets.append(offset)
                        return stream.seek(offset)

                    def read(self, size):
                        if len(observations) == 1:
                            size = min(size, max(0, 137 - stream.tell()))
                        return stream.read(size)

                yield Source()

        monkeypatch.setattr(http.artifacts, "open", opened)
        directory = tmp_path / "resumed"
        with pytest.raises((ProtocolError, TransportError)):
            saved.download(directory, resume=True).result(10)
        assert not (directory / "manifest.json").exists()
        assert saved.download(directory, resume=True).result(10) == directory
        assert observations == [[0], [137]]
        status = training.requests.get(
            saved.manifest().result(5).info.request_id
        ).result(5)
        assert status.weights == saved and status.outcome == "completed"
        with (directory / "ranks/rank-00000.ssp").open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == rank.sha256


def test_upload_of_weights_manifest_requires_exact_committed_ranks(tmp_path):
    with connected(tmp_path, weights=True) as (training, _, _, *_):
        saved = training.export_weights().result(10)
        directory = saved.download(tmp_path / "upload-source").result(10)
        training.close().result(10)
    other = tmp_path / "empty"
    other.mkdir()
    with (
        endpoint(other) as http,
        ServiceClient(address(http.client), token=TOKEN) as service,
    ):
        with pytest.raises(Exception, match="invalid_request"):
            service.artifacts.upload(
                directory / "manifest.json", reference=saved.reference
            ).result(10)
        assert http.artifacts.get(saved.id) is None
        uploaded = service.artifacts.upload(
            directory, reference=saved.reference
        ).result(10)
        assert uploaded.reference == saved.reference
