"""Ordered native restore through real CPU children, HTTP and durable records."""

import io
import sys
from dataclasses import replace
from threading import Event

import pytest
import torch

from engine_tests.execution.test_checkpoint import MemoryExecutor
from engine_tests.server.test_artifacts import reference, store
from engine_tests.server.test_checkpoint_client import connected
from engine_tests.server.test_checkpoints import stream_rank
from engine_tests.server.test_checkpoints import workers as saved_workers
from engine_tests.server.test_request_records import request
from engine_tests.server.test_request_records import store as records_store
from engine_tests.training.test_prepared_requests import family
from engine_tests.training.test_worker_checkpoints import dispatch, running, save_call
from shadowspill.checkpoint import checkpoint_size, load_checkpoint
from shadowspill.checkpoint.identity import state_identity
from shadowspill.orchestration.launch.file_grants import resolve_file
from shadowspill.orchestration.launch.staging import FileDelivery
from shadowspill.server.checkpoint_restore import CheckpointResolver
from shadowspill.server.checkpoints import publish_checkpoint
from shadowspill.training.session.acceptance import AcceptanceOrder
from shadowspill.training.session.checkpoint import CHECKPOINT_METADATA_BYTES
from shadowspill.training.session.controls import CallControls
from shadowspill.training.session.dispatch import RequestDispatcher
from shadowspill.training.session.prepared import PreparedCall
from shadowspill.training.session.state import OperationState
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.checkpoints import SaveInfo
from shadowspill_client.protocol.codec import decode, digest, encode
from shadowspill_client.protocol.requests import OperationRequest
from shadowspill_client.results import TrainingRequestError


def configuration():
    return family().inventory.layouts.setup.config


def generation(
    artifacts, *, count=2, updates=3, name="source", bad_rank=False, config=None
):
    config_ref = artifacts.get("config")
    if config is not None:
        body = encode(config)
        config_ref = reference(body, "configuration." + digest(config))
        artifacts.put(config_ref, lambda _: io.BytesIO(body))
    info = SaveInfo(
        "session",
        "run",
        "saved-history",
        f"{name}-{updates}",
        updates,
        count,
        config_ref,
        artifacts.get("plans"),
        sys.byteorder,
        metadata={"cursor": [4, 8]},
    )
    executors, ranks = saved_workers(info)
    for executor in executors:
        executor.values["bf16"].add_(10)
    if bad_rank:
        executors[-1].values["bf16"] = torch.ones(2, dtype=torch.bfloat16)
        ranks[-1] = replace(
            ranks[-1],
            state_sha256=state_identity(executors[-1]),
            size_bytes=checkpoint_size(executors[-1], metadata=info.rank_metadata()),
        )
    published = publish_checkpoint(
        artifacts,
        info,
        ranks,
        lambda rank, sink: stream_rank(executors[rank.rank], info, sink),
        name=name,
    )
    return published, executors


def compare_saved(artifacts, reference, expected, root):
    with artifacts.open(reference) as stream:
        manifest = decode(stream.read())
    for row, original in zip(manifest.ranks, expected, strict=True):
        destination = MemoryExecutor()
        destination.logical_rank = row.rank
        path = root / f"verify-{row.rank}.ssp"
        with artifacts.open(row.archive) as source:
            path.write_bytes(source.read())
        load_checkpoint(destination, path, chunk_bytes=24)
        for name, value in original.values.items():
            torch.testing.assert_close(destination.values[name].float(), value.float())
    return manifest


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "geometry",
        "delivery",
        "no_loader",
        "small_staging",
        "load_failure",
        "lost_restore",
        "publication",
    ],
)
def test_all_rank_restore_validates_before_release_and_never_replays(
    tmp_path, monkeypatch, fault
):
    with (
        store(tmp_path) as artifacts,
        records_store(tmp_path) as records,
        running(tmp_path, artifacts, mode=fault or "normal") as (
            workers,
            publisher,
            storage,
        ),
    ):
        source, expected = generation(artifacts, bad_rank=fault == "geometry")
        state = OperationState(completed_updates=12, history_id="initial-history")
        before = state.boundary
        dispatcher = RequestDispatcher(records, state, max_pending=4)
        command = OperationRequest(
            replace(request().identity, request_id="load"), "restore", restore="source"
        )
        budget = BufferBudget(CHECKPOINT_METADATA_BYTES)
        call = PreparedCall(
            command,
            None,
            None,
            CallControls(),
            None,
            budget,
            ranks=(0, 1),
            _workspace=budget.reserve(CHECKPOINT_METADATA_BYTES),
        )
        resolver = CheckpointResolver(
            artifacts, session_id="session", world_size=2, configuration=configuration()
        )

        def resolve():
            call.restore = resolver(command)
            return call.restore.reference

        records.submit(command)
        original_digest = digest(command)
        accepted = dispatcher.order.validated("load", checkpoint=resolve)
        assert accepted.acceptance.checkpoint == source
        assert digest(records.get("load")[0]) == original_digest
        if fault == "delivery":
            opened = storage.open
            bad = call.restore.manifest.ranks[-1].archive
            monkeypatch.setattr(
                storage,
                "open",
                lambda ref: io.BytesIO(b"bad") if ref == bad else opened(ref),
            )

        def publish(receipt, work):
            assert records.execution("load") == receipt
            if fault == "publication":
                raise OSError("publication unavailable")
            return publisher(call, receipt, work)

        def execute(transition):
            return workers.execute(call, transition)

        outcome = dispatcher.run("load", execute, publish, restore=call.restore)
        assert dispatcher.run("load", execute, publish, restore=call.restore) == outcome
        if fault in ("geometry", "delivery", "no_loader", "small_staging"):
            assert outcome.failure.execution_effect == "not_started"
            assert state.boundary == before and workers.group.valid
            assert not list(tmp_path.glob("executions-*"))
        elif fault in ("load_failure", "lost_restore"):
            assert outcome.failure.execution_effect == "unknown"
            assert not state.boundary.valid and not workers.group.valid
            assert records.execution("load") is None
        else:
            assert state.boundary.valid and state.boundary.completed_updates == 3
            assert state.boundary.history_id != before.history_id
            assert records.execution("load").history_id == state.boundary.history_id
            if fault == "publication":
                assert outcome.failure.execution_effect == "completed"
                assert outcome.failure.history_id == state.boundary.history_id
            else:
                assert outcome.result.checkpoint == source
                assert outcome.result.metadata["cursor"] == (4, 8)
            following = dispatch(records, dispatcher, workers, publisher, save_call(2))
            assert following.state == "completed"
            manifest = compare_saved(
                artifacts, following.result.checkpoint, expected, tmp_path
            )
            assert manifest.info.history_id == state.boundary.history_id
            assert manifest.info.completed_updates == 3
            assert [
                (tmp_path / f"executions-{rank}").read_text() for rank in range(2)
            ] == ["1\n2\n", "1\n2\n"]
        call.close()
        assert budget.used == (0, 0)


def test_public_load_pins_alias_and_returns_saved_metadata_before_later_save(
    tmp_path, monkeypatch
):
    with connected(tmp_path, completed_updates=12) as (
        training,
        _,
        http,
        records,
        controller,
        workers,
    ):
        source, expected = generation(http.artifacts, count=1)
        entered, release = Event(), Event()
        execute = workers.execute

        def delayed(call, transition):
            if call.request.operation == "restore":
                entered.set()
                assert release.wait(10)
            return execute(call, transition)

        monkeypatch.setattr(workers, "execute", delayed)
        pending = training.load_checkpoint("source")
        try:
            accepted = pending.accepted().result(10)
            assert accepted.checkpoint == source and entered.wait(5)
            # Moving the alias after acceptance cannot replace this request's input.
            newer, _ = generation(http.artifacts, count=1, name="source", updates=8)
            assert newer != source
            assert http.artifacts.resolve_alias("session", "run", "source") == newer
            following = training.save_checkpoint()
        finally:
            release.set()
        result = pending.result(10)
        assert result.metadata["cursor"] == (4, 8)
        assert result.completed_updates == 3 and result.history_id != "history"
        assert result.checkpoint.reference == source
        saved = following.result(10)
        manifest = compare_saved(http.artifacts, saved.reference, expected, tmp_path)
        assert manifest.info.history_id == result.history_id
        assert records.get(pending.request_id)[0].restore == "source"
        assert (
            training.requests.get(pending.request_id).result(5).record == result.record
        )
        assert training.close().result(10).completed_updates == 3
        assert controller.boundary.closed and controller.boundary.valid


def test_public_missing_checkpoint_rejects_before_worker_work(tmp_path):
    with connected(tmp_path) as (training, _, _, records, controller, _):
        pending = training.load_checkpoint("missing")
        with pytest.raises(TrainingRequestError) as failed:
            pending.result(10)
        assert failed.value.failure.execution_effect == "not_started"
        assert records.get(pending.request_id)[1].acceptance is None
        assert controller.boundary.valid and not list(tmp_path.glob("executions-*"))
        assert training.close().result(10).completed_updates == 0


def test_public_restore_reuses_startup_grant_without_copying_or_retiring_it(
    tmp_path, monkeypatch
):
    with connected(tmp_path, completed_updates=12) as (
        training,
        _,
        http,
        _,
        controller,
        workers,
    ):
        source, expected = generation(http.artifacts, count=1)
        with http.artifacts.open(source) as stream:
            manifest = decode(stream.read())
        reference = manifest.ranks[0].archive
        node = workers.nodes[0]
        with FileDelivery(
            [reference], root=node.root, disk=node.disk, memory=node.memory
        ) as delivery:
            delivery.receive(http.artifacts.open)
            with node.grants.lend(delivery, [reference]):
                before = node.disk.used

                def copied_again(*args):
                    pytest.fail("startup checkpoint was copied into another delivery")

                monkeypatch.setattr(workers._checkpoints, "open", copied_again)
                histories = set()
                for _ in range(2):
                    restored = training.load_checkpoint(source.id).result(10)
                    assert restored.completed_updates == 3
                    histories.add(restored.history_id)
                    assert controller.boundary.valid
                    assert resolve_file(node.grants.root, reference).is_file()
                    assert len(node.grants._owners) == 1 and node.disk.used == before
                assert len(histories) == 2
                saved = training.save_checkpoint().result(10)
                compare_saved(http.artifacts, saved.reference, expected, tmp_path)
                with pytest.raises(RuntimeError, match="consumers"):
                    delivery.close()
        assert not node.grants._owners
        assert training.close().result(10).completed_updates == 3


@pytest.mark.parametrize(
    "difference", ["semantics", "definition", "source", "defaults", "objective"]
)
def test_ordered_restore_checks_optimizer_interpretation_before_acceptance(
    tmp_path, difference
):
    with connected(tmp_path, completed_updates=12) as (
        training,
        _,
        http,
        records,
        controller,
        _,
    ):
        config = training.get_config()
        if difference == "semantics":
            changed = replace(
                config,
                optimizer=replace(config.optimizer, config={"reinterpret": True}),
            )
        elif difference == "definition":
            changed = replace(
                config,
                optimizer=replace(config.optimizer, definition="other-optimizer"),
            )
        elif difference == "source":
            changed = replace(
                config,
                definition_sources={
                    **config.definition_sources,
                    "optimizer": replace(
                        config.definition_sources["optimizer"], version="different"
                    ),
                },
            )
        elif difference == "defaults":
            changed = replace(
                config, optimizer=replace(config.optimizer, defaults={"lr": 100.0})
            )
        else:
            changed = replace(
                config,
                objective=replace(config.objective, definition="another-objective"),
            )
        source, _ = generation(http.artifacts, count=1, config=changed)
        before = controller.boundary
        load = training.load_checkpoint(source.id)
        if difference in ("semantics", "definition", "source"):
            with pytest.raises(TrainingRequestError, match="optimizer semantics"):
                load.result(10)
            assert records.get(load.request_id)[1].acceptance is None
            assert controller.boundary == before
            assert not list(tmp_path.glob("executions-*"))
        else:
            result = load.result(10)
            assert (
                result.completed_updates == 3 and result.history_id != before.history_id
            )
            assert training.get_config() == config
        training.close().result(10)


def test_alias_resolves_at_ordered_acceptance_and_survives_record_reopen(tmp_path):
    with store(tmp_path) as artifacts, records_store(tmp_path) as records:
        for name in ("config", "plans"):
            body = encode(configuration()) if name == "config" else name.encode()
            artifacts.put(reference(body, name), lambda _, body=body: io.BytesIO(body))
        original, _ = generation(artifacts, count=1)
        resolver = CheckpointResolver(
            artifacts, session_id="session", world_size=1, configuration=configuration()
        )
        command = OperationRequest(
            replace(request().identity, request_id="load", sequence=2),
            "restore",
            restore="source",
        )
        records.submit(request("first", 1))
        records.submit(command)
        order = AcceptanceOrder(records, OperationState(), max_pending=2)
        resolved = []

        def resolve():
            result = resolver(command)
            resolved.append(result.reference)
            return result.reference

        assert order.validated("load", checkpoint=resolve).state == "uploading"
        assert not resolved
        latest, _ = generation(artifacts, count=1, updates=8)
        assert latest != original
        order.validated("first")
        accepted = records.get("load")[1]
        assert accepted.acceptance.checkpoint == latest and resolved == [latest]
        generation(artifacts, count=1, updates=12)
        assert order.validated("load", checkpoint=resolve) == accepted
        assert resolved == [latest]
        assert records.submit(command) == accepted
    with records_store(tmp_path) as reopened:
        assert reopened.get("load") == (command, accepted)


@pytest.mark.parametrize("difference", ["world_size", "dependency"])
def test_incompatible_generation_fails_before_acceptance(tmp_path, difference):
    with connected(tmp_path) as (_, _, http, _, _, _):
        source, _ = generation(http.artifacts, count=1)
        resolver = CheckpointResolver(
            http.artifacts,
            session_id="session",
            world_size=2 if difference == "world_size" else 1,
            configuration=configuration(),
        )
        command = OperationRequest(request().identity, "restore", restore=source.id)
        if difference == "dependency":
            # Dependency access is independently owner-scoped; the manifest is
            # insufficient when the source Plans are unavailable in this namespace.
            get = http.artifacts.get
            from unittest.mock import patch

            with (
                patch.object(
                    http.artifacts,
                    "get",
                    lambda key: None if key == "plans" else get(key),
                ),
                pytest.raises(ValueError, match="dependency"),
            ):
                resolver(command)
        else:
            with pytest.raises(ValueError, match="conversion"):
                resolver(command)
