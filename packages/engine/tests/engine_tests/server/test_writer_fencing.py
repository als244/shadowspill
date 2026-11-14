"""Writer takeover fences only new work, with durable, idempotent claim identity."""

from dataclasses import replace

import pytest

from engine_tests.server.test_request_records import IDENTITY, request, result, store
from shadowspill.server.records.requests import RequestRecords


def test_takeover_fences_pending_and_new_work_but_preserves_accepted_work(tmp_path):
    with store(tmp_path) as records:
        accepted, pending = request("a", 1), request("b", 2)
        records.submit(accepted)
        records.accept("a")
        records.submit(pending)
        lease = records.claim_writer("writer2", "server")
        assert lease.writer_epoch == 2 and lease.next_sequence == 3
        with pytest.raises(PermissionError, match="fenced"):
            records.accept("b")
        with pytest.raises(PermissionError, match="fenced"):
            records.submit(request("old", 3))
        current = request("new", 3)
        current = replace(
            current,
            identity=replace(
                current.identity,
                writer_id=lease.writer_id,
                writer_epoch=lease.writer_epoch,
            ),
        )
        assert records.submit(current).state == "uploading"
        records.dispatch("a")
        completed = records.complete(result(accepted))
        assert records.submit(accepted) == completed
        # Repeating a pending upload retrieves its status but cannot accept it.
        assert records.submit(pending).state == "uploading"
    with store(tmp_path) as reopened:
        assert reopened.writer == lease
        assert reopened.get("a")[1] == completed
        with pytest.raises(PermissionError, match="fenced"):
            reopened.accept("b")
        assert reopened.accept("new").state == "accepted"


def test_lost_claim_response_reuses_original_lease_and_cannot_reclaim_a_retired_id(
    tmp_path,
):
    with store(tmp_path) as records:
        first = records.writer
        assert records.claim_writer("writer", "server") == first
        records.submit(request())
        second = records.claim_writer("second", "server")
        assert second.writer_epoch == 2 and second.next_sequence == 2
        assert records.claim_writer("second", "server") == second
        with pytest.raises(PermissionError, match="retired"):
            records.claim_writer("writer", "server")
        with pytest.raises(PermissionError, match="retired"):
            records.claim_writer("second", "another_server")
    with store(tmp_path) as reopened:
        assert reopened.claim_writer("second", "server") == second
        with pytest.raises(PermissionError, match="retired"):
            reopened.claim_writer("writer", "server")


def test_unclaimed_run_and_wrong_incarnation_never_accept_new_requests(tmp_path):
    with RequestRecords(
        tmp_path / "journal", identity=IDENTITY, cache_root=tmp_path
    ) as records:
        with pytest.raises(PermissionError, match="absent"):
            records.submit(request())
        records.claim_writer("writer", "server")
        wrong = request()
        wrong = replace(
            wrong, identity=replace(wrong.identity, server_incarnation="other")
        )
        with pytest.raises(PermissionError, match="fenced"):
            records.submit(wrong)
        assert records.submit(request()).state == "uploading"
