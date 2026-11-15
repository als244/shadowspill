"""Persistent files remain owned across the sole dispatcher's receipt boundary."""

from dataclasses import replace

import pytest

from engine_tests.server.test_artifact_writer import encode, spec
from engine_tests.server.test_artifacts import reference, store
from engine_tests.server.test_request_records import store as request_store
from engine_tests.training.test_request_dispatch import publish, submit
from shadowspill.server.artifact_writer import ProducedArtifacts
from shadowspill.server.records.artifact_index import ArtifactAlias
from shadowspill.training.session.dispatch import RequestDispatcher
from shadowspill.training.session.state import OperationState
from shadowspill_client.buffers import BufferBudget


def stage(artifacts):
    return artifacts.stage_group(
        [spec(reference())],
        lambda _, writer: encode(writer),
        alias=ArtifactAlias("session", "run", "latest", reference().id),
    )


@pytest.mark.parametrize("commit", [False, True])
def test_completed_production_is_private_and_retains_storage_until_owner_closes(
    tmp_path, commit
):
    memory = BufferBudget(8 << 20, 10)
    with store(tmp_path, memory=memory) as artifacts:
        with stage(artifacts) as group:
            assert group.references == (reference(),)
            assert artifacts.get(reference().id) is None
            assert artifacts.resolve_alias("session", "run", "latest") is None
            assert artifacts.used == (6, 1) and artifacts._disk.used == (6, 1)
            with pytest.raises(RuntimeError, match="producers"):
                artifacts.close()
            position = artifacts._journal.position
            if commit:
                assert group.commit() == (reference(),)
                assert artifacts._journal.position == position + 2
                assert group.commit() == (reference(),)
                assert artifacts._journal.position == position + 2
            assert not group.closed
        assert group.closed and artifacts._disk.used == (0, 0)
        assert artifacts.used == ((6, 1) if commit else (0, 0))
        group.close()
        with pytest.raises(RuntimeError, match="closed"):
            group.commit()
    assert memory.used == (0, 0)
    with store(tmp_path) as reopened:
        assert reopened.resolve_alias("session", "run", "latest") == (
            reference() if commit else None
        )


@pytest.mark.parametrize("retry_owner", [False, True])
def test_failed_close_can_be_retried_by_owner_or_storage_without_double_release(
    tmp_path, monkeypatch, retry_owner
):
    memory = BufferBudget(8 << 20, 10)
    artifacts = store(tmp_path, memory=memory)
    group = stage(artifacts)
    close = ProducedArtifacts.close

    def fail(_):
        raise OSError("temporary files still busy")

    monkeypatch.setattr(ProducedArtifacts, "close", fail)
    with pytest.raises(OSError):
        group.close()
    assert not group.closed and artifacts.used == (6, 1)
    assert len(artifacts._pending) == 1
    monkeypatch.setattr(ProducedArtifacts, "close", close)
    if retry_owner:
        group.close()
    artifacts.close()
    group.close()
    assert group.closed and memory.used == (0, 0)


@pytest.mark.parametrize("failure", [None, "receipt", "before_commit", "after_commit"])
def test_dispatcher_receipt_precedes_publication_and_failures_never_reexecute(
    tmp_path, monkeypatch, failure
):
    owners, invocations = [], []
    with store(tmp_path) as artifacts, request_store(tmp_path) as records:
        state = OperationState()
        dispatcher = RequestDispatcher(records, state, max_pending=1)
        submit(dispatcher, records, "a", 1)
        executed, append = records._index.executed, artifacts._journal.append

        def fail_receipt(receipt):
            executed(receipt)
            raise OSError("receipt index acknowledgement failed")

        def fail_publication(entries):
            if failure == "after_commit":
                append(entries)
            raise OSError("artifact commit acknowledgement failed")

        if failure == "receipt":
            monkeypatch.setattr(records._index, "executed", fail_receipt)
        elif failure is not None:
            monkeypatch.setattr(artifacts._journal, "append", fail_publication)

        def execute(transition):
            assert records.get("a")[1].state == "running"
            assert records.execution("a") is None
            invocations.append(transition.request_id)
            group = stage(artifacts)
            owners.append(group)
            return group

        def output(receipt, group):
            assert records.execution("a") == receipt
            assert receipt.completed_updates == 1
            assert artifacts.get(reference().id) is None
            refs = group.commit()
            return replace(publish(receipt, group), outputs={"saved": refs[0]})

        if failure == "receipt":
            with pytest.raises(OSError, match="receipt index"):
                dispatcher.run("a", execute, output, participates=True)
            assert not state.boundary.valid
        else:
            status = dispatcher.run("a", execute, output, participates=True)
            assert state.boundary.valid and state.boundary.completed_updates == 1
            if failure is None:
                assert status.result.outputs == {"saved": reference()}
            else:
                assert status.failure.execution_effect == "completed"
                assert status.failure.completed_updates == 1
            assert dispatcher.run("a", execute, output, participates=True) == status
        assert invocations == ["a"] and owners[0].closed
        assert artifacts._disk.used == (0, 0) and not artifacts._pending
    with store(tmp_path) as recovered, request_store(tmp_path) as records:
        expected = reference() if failure in (None, "after_commit") else None
        assert recovered.get(reference().id) == expected
        assert recovered.resolve_alias("session", "run", "latest") == expected
        assert records.execution("a").completed_updates == 1
