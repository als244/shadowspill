"""Lost replies and index failure never create a second accepted execution."""

import subprocess
import sys
from dataclasses import replace

import pytest

from shadowspill.server.records import Journal
from shadowspill.server.records.requests import RequestRecords, RunIdentity
from shadowspill_client.protocol.requests import (
    ArtifactRef,
    OperationRequest,
    RequestIdentity,
)
from shadowspill_client.protocol.results import (
    Acceptance,
    ExecutionReceipt,
    Failure,
    OperationResult,
    RequestStatus,
)

IDENTITY = RunIdentity("owner", "session", "run")


def request(name="a", sequence=1):
    return OperationRequest(
        RequestIdentity(name, "run", sequence, "writer", 1, "server"), "forward"
    )


def result(command):
    return OperationResult(
        command.identity,
        command.operation,
        "history",
        "completed",
        0,
        metrics={"count": 3},
    )


def store(root, *, identity=IDENTITY):
    records = RequestRecords(root / "journal", identity=identity, cache_root=root)
    if records.writer is None:
        records.claim_writer("writer", "server")
    return records


def test_request_and_result_deduplication_survive_reopen_and_index_rebuild(tmp_path):
    command = request()
    with store(tmp_path) as records:
        submitted = records.submit(command)
        assert submitted.state == "uploading" and submitted.acceptance is None
        accepted = records.accept("a")
        assert accepted.acceptance.journal_position == 4  # run, writer, submit, accept
        assert records.submit(command) == accepted
        records.dispatch("a")
        with pytest.raises(ValueError, match="never replay"):
            records.dispatch("a")
        records.executed(ExecutionReceipt.from_result(result(command)))
        completed = records.complete(result(command))
        assert records.complete(result(command)) == completed
        assert not records.pending()
        assert records.accept("a").acceptance == accepted.acceptance
    assert not list(tmp_path.glob("request-index-*"))
    with store(tmp_path) as reopened:
        assert reopened.get("a") == (command, completed)
        assert reopened.submit(command) == completed
        assert reopened.last_sequence == 1
        with pytest.raises(ValueError, match="immutable"):
            reopened.fail("a", Failure("bad", "late error", "not_started", True))


def test_conflicting_ids_sequences_and_payload_checksums_cannot_replace_requests(
    tmp_path,
):
    command = request()
    with store(tmp_path) as records:
        records.submit(command)
        for changed in (
            replace(command, outputs="different"),
            replace(command, payload=ArtifactRef("payload", "a" * 64, 1)),
            replace(command, identity=replace(command.identity, sequence=2)),
        ):
            with pytest.raises(ValueError, match="different content"):
                records.submit(changed)
        with pytest.raises(ValueError, match="sequence already"):
            records.submit(request("b", 1))
        # Identical work under a new request ID/sequence is an intentional call.
        assert records.submit(request("b", 2)).identity.request_id == "b"
        assert records.get("a")[0] == command
        with pytest.raises(ValueError, match="another run"):
            records.submit(
                replace(command, identity=replace(command.identity, run_id="other"))
            )


def test_out_of_order_upload_records_are_queryable_in_submission_order(tmp_path):
    with store(tmp_path) as records:
        for index in (4, 2, 1, 3):
            records.submit(request(str(index), index))
        assert [s.identity.sequence for s in records.pending(limit=2)] == [1, 2]
        assert [s.identity.sequence for s in records.pending(after=2, limit=2)] == [
            3,
            4,
        ]
        records.fail("2", Failure("invalid", "bad input", "not_started", True))
        assert [s.identity.sequence for s in records.pending()] == [1, 3, 4]
        with pytest.raises(ValueError, match=r"1\.\.128"):
            records.pending(limit=1000)


@pytest.mark.parametrize("state", ["uploading", "accepted", "running"])
def test_restart_retains_unfinished_status_without_dispatch_or_replay(tmp_path, state):
    command = request()
    with store(tmp_path) as records:
        records.submit(command)
        if state != "uploading":
            records.accept("a")
        if state == "running":
            records.dispatch("a")
    with store(tmp_path) as recovered:
        assert recovered.get("a")[1].state == state
        assert recovered.submit(command).state == state
        # The coordinator must explicitly classify worker loss; the store only
        # persists its factual execution effect and immutable failure.
        failure = Failure(
            "restart",
            "live workers were lost",
            "unknown" if state == "running" else "not_started",
            False,
        )
        terminal = recovered.fail("a", failure)
        assert terminal.failure == failure and not recovered.pending()


def test_known_execution_and_output_publication_failure_remain_distinct(tmp_path):
    with store(tmp_path) as records:
        command = request()
        records.submit(command)
        records.accept("a")
        with pytest.raises(ValueError, match="undispatched"):
            records.fail("a", Failure("output", "failed output", "completed", True, 1))
        records.dispatch("a")
        receipt = ExecutionReceipt(
            command.identity, command.operation, "history", "completed", 1
        )
        records.executed(receipt)
        failure = Failure(
            "output",
            "failed output",
            "completed",
            True,
            1,
            receipt.accumulation,
            history_id=receipt.history_id,
        )
        failed = records.fail("a", failure)
        assert records.submit(command) == failed
        assert failed.failure.execution_effect == "completed"
        with pytest.raises(ValueError, match="never replay"):
            records.dispatch("a")


def test_index_failure_after_durable_completion_requires_reconciliation(
    tmp_path, monkeypatch
):
    records = store(tmp_path)
    command = request()
    records.submit(command)
    records.accept("a")
    records.dispatch("a")
    records.executed(ExecutionReceipt.from_result(result(command)))

    def fail_index(status, *, position):
        raise OSError("query cache unavailable")

    monkeypatch.setattr(records._index, "update", fail_index)
    with pytest.raises(OSError, match="query cache unavailable"):
        records.complete(result(command))
    with pytest.raises(RuntimeError, match="reopen and reconcile"):
        records.submit(request("b", 2))
    records.close()
    with store(tmp_path) as recovered:
        assert recovered.get("a")[1].result == result(command)
        assert not recovered.pending()


def test_owner_mismatch_does_not_mutate_journal_or_leak_its_lock(tmp_path):
    with store(tmp_path) as records:
        records.submit(request())
    files = {p.name: p.read_bytes() for p in (tmp_path / "journal").glob("*.ssj")}
    with pytest.raises(ValueError, match="another owner/session/run"):
        store(tmp_path, identity=replace(IDENTITY, owner_id="different"))
    assert files == {
        p.name: p.read_bytes() for p in (tmp_path / "journal").glob("*.ssj")
    }
    with store(tmp_path) as records:
        assert records.last_sequence == 1


def test_replay_rejects_a_semantically_invalid_but_checksummed_acceptance(tmp_path):
    command = request()
    with store(tmp_path) as records:
        records.submit(command)
    with Journal(tmp_path / "journal") as journal:
        status = RequestStatus(
            command.identity, "accepted", Acceptance(command.identity, 99)
        )
        journal.append([("request.status", status.to_dict())])
    with pytest.raises(ValueError, match="different durable journal position"):
        store(tmp_path)


def test_record_layer_has_no_framework_runtime_or_catalog_dependency(tmp_path):
    code = """
import sys
from pathlib import Path
for name in ('torch', 'triton', 'mlops', 'shadowspill_workloads',
             'shadowspill.runtime', 'shadowspill.training'):
    sys.modules[name] = None
from shadowspill.server.records.requests import RequestRecords, RunIdentity
from shadowspill_client.protocol.requests import OperationRequest, RequestIdentity
root = Path(sys.argv[1])
with RequestRecords(root/'journal', identity=RunIdentity('o','s','r'),
                    cache_root=root) as records:
    records.claim_writer('w', 'server')
    command = OperationRequest(RequestIdentity('a','r',1,'w',1,'server'), 'forward')
    records.submit(command)
    assert records.accept('a').acceptance.journal_position == 4
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
