"""Real child restore delivers a new base artifact and changes later save provenance."""

import io
from dataclasses import replace

import pytest
import torch

from engine_tests.server.test_artifacts import reference, store
from engine_tests.server.test_checkpoint_restore import configuration
from engine_tests.server.test_checkpoints import stream_rank
from engine_tests.server.test_request_records import request
from engine_tests.server.test_request_records import store as records_store
from engine_tests.training.test_worker_checkpoints import dispatch, running, save_call
from engine_tests.training.test_worker_frozen_checkpoint import fixture, info
from shadowspill.checkpoint import checkpoint_size
from shadowspill.checkpoint.identity import state_identity
from shadowspill.server.checkpoint_restore import (
    CheckpointResolver,
    require_checkpoint_dependencies,
)
from shadowspill.server.checkpoints import publish_checkpoint
from shadowspill.training.session.dispatch import RequestDispatcher
from shadowspill.training.session.state import OperationState
from shadowspill.training.worker.checkpoint_records import RankRestore, RankSave
from shadowspill_client import Weights
from shadowspill_client.protocol.codec import decode, encode
from shadowspill_client.protocol.requests import OperationRequest


def generation(artifacts, value, base, name):
    boundary = replace(
        info(base, request=name),
        configuration=artifacts.get("config"),
        plans=artifacts.get("plans"),
    )
    rank = RankSave(
        0,
        checkpoint_size(
            value,
            metadata=boundary.rank_metadata(),
            frozen_state_id=boundary.frozen_state_id,
        ),
        state_identity(value),
        value.plan.program.digest,
        {},
    )
    return publish_checkpoint(
        artifacts, boundary, [rank], lambda _, sink: stream_rank(value, boundary, sink)
    )


def test_ordered_restore_delivers_base_and_later_saves_adopt_or_clear_it(tmp_path):
    with (
        store(tmp_path) as artifacts,
        records_store(tmp_path) as records,
        running(tmp_path, artifacts, count=1, mode="frozen") as (
            workers,
            publisher,
            storage,
        ),
    ):
        value, _, base, reader, files, _ = fixture(tmp_path)
        storage.frozen_base = base  # Same admitted initialization as the child.
        dispatcher = RequestDispatcher(records, OperationState(), max_pending=4)
        first = dispatch(
            records, dispatcher, workers, publisher, replace(save_call(), ranks=(0,))
        )
        assert first.state == "completed", first.failure
        with artifacts.open(first.result.checkpoint) as stream:
            assert decode(stream.read()).info.frozen_base == base
        weights = reference(b"100", "other-base")
        artifacts.put(weights, lambda _: io.BytesIO(b"100"))
        base = replace(
            base,
            model=replace(base.model, initialize=Weights.artifact(weights.id)),
            artifact=weights,
        )
        value.values["base"].add_(100)
        partial = generation(artifacts, value, base, "partial-source")
        value.values["base"].add_(25)
        complete = generation(artifacts, value, None, "complete-source")
        resolver = CheckpointResolver(
            artifacts, session_id="session", world_size=1, configuration=configuration()
        )
        for index, source in enumerate((partial, complete)):
            sequence = 2 + 2 * index
            command = OperationRequest(
                replace(
                    request().identity,
                    request_id=f"restore-{sequence}",
                    sequence=sequence,
                ),
                "restore",
                restore=source.id,
            )
            call = replace(save_call(sequence), request=command, ranks=(0,))
            call.restore = resolver(command)
            records.submit(command)
            dispatcher.order.validated(
                command.identity.request_id, checkpoint=lambda source=source: source
            )
            outcome = dispatcher.run(
                command.identity.request_id,
                lambda transition, call=call: workers.execute(call, transition),
                lambda receipt, work, call=call: publisher(call, receipt, work),
                restore=call.restore,
            )
            assert outcome.state == "completed", outcome.failure
            call.close()
            expected_base = base if index == 0 else None
            assert storage.frozen_base == expected_base
            following = dispatch(
                records,
                dispatcher,
                workers,
                publisher,
                replace(save_call(sequence + 1), ranks=(0,)),
            )
            assert following.state == "completed", following.failure
            with artifacts.open(following.result.checkpoint) as stream:
                manifest = decode(stream.read())
            assert manifest.info.frozen_base == expected_base
            row = manifest.ranks[0]
            path = tmp_path / f"verify-{index}.ssp"
            with artifacts.open(row.archive) as stream:
                path.write_bytes(stream.read())
            files[row.archive] = path
            files[weights] = tmp_path / "other-base"
            files[weights].write_bytes(b"100")
            target = RankRestore(following.result.checkpoint, manifest.info, row)
            reader.load(target, reader.prepare_source(target))
            torch.testing.assert_close(
                value.values["base"],
                torch.arange(6).float() + (100 if index == 0 else 125),
            )
        assert not workers.nodes[0].grants._entries


def test_manifest_registration_does_not_copy_base_but_restore_requires_exact_dependency(
    tmp_path, monkeypatch
):
    with store(tmp_path) as artifacts:
        value, _, base, _, _, _ = fixture(tmp_path)
        body = encode(configuration())
        artifacts.put(reference(body, "config"), lambda _: io.BytesIO(body))
        artifacts.put(reference(b"plans", "plans"), lambda _: io.BytesIO(b"plans"))
        weights = reference(b"0", "base-weights")
        base = replace(
            base,
            model=replace(base.model, initialize=Weights.artifact(weights.id)),
            artifact=weights,
        )
        with pytest.raises(ValueError, match="dependency is not committed"):
            generation(artifacts, value, base, "partial")
        artifacts.put(weights, lambda _: io.BytesIO(b"0"))
        checkpoint = generation(artifacts, value, base, "partial")
        resolver = CheckpointResolver(
            artifacts, session_id="session", world_size=1, configuration=configuration()
        )
        resolved = resolver.resolve(checkpoint)
        get = artifacts.get
        monkeypatch.setattr(
            artifacts,
            "get",
            lambda identity: None if identity == weights.id else get(identity),
        )
        require_checkpoint_dependencies(artifacts, resolved.manifest, setup=False)
        with pytest.raises(ValueError, match="dependency is unavailable: base-weights"):
            resolver.resolve(checkpoint)
