"""Durable restart outcomes do not imply rollback or automatic mutation replay."""

import pytest

from engine_tests.server.test_request_records import request, result, store
from shadowspill.training.session.recovery import fail_interrupted_requests
from shadowspill.training.session.state import OperationState


def prepare(records):
    for index, name in enumerate(("done", "running", "accepted", "upload"), 1):
        command = request(name, index)
        records.submit(command)
        if name == "upload":
            continue
        records.accept(name)
        if name == "accepted":
            continue
        records.dispatch(name)
        if name == "done":
            records.complete(result(command))


def test_restart_classifies_each_boundary_and_preserves_completed_results(tmp_path):
    with store(tmp_path) as records:
        prepare(records)
        completed = records.get("done")[1]
    with store(tmp_path) as recovered:
        state = OperationState()
        report = fail_interrupted_requests(recovered, state)
        assert (report.failed_requests, report.uncertain_executions) == (3, 1)
        assert not state.boundary.valid
        assert recovered.get("done")[1] == completed
        for name in ("running", "accepted", "upload"):
            _, status = recovered.get(name)
            assert status.state == "failed" and not status.failure.state_valid
            assert status.failure.completed_updates is None
            assert status.failure.accumulation is None
            assert status.failure.execution_effect == (
                "unknown" if name == "running" else "not_started"
            )
            assert (status.acceptance is not None) == (name != "upload")
        with pytest.raises(RuntimeError, match="invalid"):
            state.begin("forward", "new")
        again = fail_interrupted_requests(recovered, state)
        assert again.failed_requests == again.uncertain_executions == 0
    with store(tmp_path) as verified:
        assert not verified.pending()
        assert verified.get("running")[1].failure.execution_effect == "unknown"


def test_authoritative_failure_stops_recovery_with_live_state_already_invalid(
    tmp_path, monkeypatch
):
    with store(tmp_path) as records:
        prepare(records)
        state = OperationState()
        real = records.fail
        calls = []

        def fail_second(request_id, failure):
            calls.append(request_id)
            if len(calls) == 2:
                raise OSError("authoritative storage lost")
            return real(request_id, failure)

        monkeypatch.setattr(records, "fail", fail_second)
        with pytest.raises(OSError, match="authoritative storage lost"):
            fail_interrupted_requests(records, state)
        assert not state.boundary.valid
        assert calls == ["running", "accepted"]
    with store(tmp_path) as reopened:
        report = fail_interrupted_requests(reopened, OperationState())
        assert (report.failed_requests, report.uncertain_executions) == (2, 0)


def test_empty_or_completed_history_never_restores_live_worker_state(tmp_path):
    with store(tmp_path) as records:
        state = OperationState(completed_updates=9)
        report = fail_interrupted_requests(records, state)
        assert report.failed_requests == 0 and not state.boundary.valid
        assert state.boundary.completed_updates == 9
