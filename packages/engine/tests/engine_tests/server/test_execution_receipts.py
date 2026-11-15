"""Confirmed execution survives publication failure without implying live workers."""

from dataclasses import replace

import pytest

from engine_tests.server.test_request_records import request, result, store
from shadowspill.training.session.recovery import fail_interrupted_requests
from shadowspill.training.session.state import OperationState
from shadowspill_client.protocol.requests import ArtifactRef
from shadowspill_client.protocol.results import ExecutionReceipt, Failure


def test_execution_receipt_is_immutable_and_persists_before_output_publication(
    tmp_path,
):
    command = request()
    completed = result(command)
    receipt = ExecutionReceipt.from_result(completed)
    with store(tmp_path) as records:
        records.submit(command)
        records.accept("a")
        records.dispatch("a")
        assert records.executed(receipt) == receipt
        assert records.get("a")[1].state == "running"
        position = records._journal.position
        assert records.executed(receipt) == receipt
        assert records._journal.position == position
        with pytest.raises(ValueError, match="immutable"):
            records.executed(replace(receipt, completed_updates=9))
    with store(tmp_path) as recovered:
        assert recovered.execution("a") == receipt
        published = replace(
            completed, outputs={"predictions": ArtifactRef("out", "a" * 64, 4)}
        )
        assert recovered.complete(published).result == published
        assert recovered.executed(receipt) == receipt


def test_final_outcome_cannot_invent_or_change_execution_facts(tmp_path):
    command = request()
    receipt = ExecutionReceipt.from_result(result(command))
    with store(tmp_path) as records:
        records.submit(command)
        with pytest.raises(ValueError, match="dispatched"):
            records.executed(receipt)
        records.accept("a")
        records.dispatch("a")
        with pytest.raises(ValueError, match="confirmed execution receipt"):
            records.complete(result(command))
        with pytest.raises(ValueError, match="confirmed execution receipt"):
            records.fail(
                "a", Failure("output", "Failed publication", "completed", True, 0)
            )
        with pytest.raises(ValueError, match="differs from its request"):
            records.executed(replace(receipt, operation="forward_backward"))
        records.executed(receipt)
        for changed in (
            replace(result(command), completed_updates=1),
            replace(result(command), outcome="skipped"),
            replace(result(command), history_id="other-history"),
        ):
            with pytest.raises(ValueError, match="differs from confirmed execution"):
                records.complete(changed)
        for failure in (
            Failure("lost", "Lost worker", "unknown", False),
            Failure(
                "output",
                "Failed publication",
                "completed",
                True,
                1,
                receipt.accumulation,
            ),
        ):
            with pytest.raises(ValueError, match="differs from confirmed execution"):
                records.fail("a", failure)
        failure = Failure(
            "output",
            "Failed publication",
            "completed",
            True,
            receipt.completed_updates,
            receipt.accumulation,
            history_id=receipt.history_id,
        )
        assert records.fail("a", failure).failure == failure


def test_receipt_index_failure_after_commit_recovers_the_original_known_effect(
    tmp_path, monkeypatch
):
    records = store(tmp_path)
    command = request()
    records.submit(command)
    records.accept("a")
    records.dispatch("a")
    receipt = ExecutionReceipt.from_result(result(command))

    def fail_index(receipt):
        raise OSError("receipt index failed")

    monkeypatch.setattr(records._index, "executed", fail_index)
    with pytest.raises(OSError, match="receipt index failed"):
        records.executed(receipt)
    with pytest.raises(RuntimeError, match="reconcile"):
        records.execution("a")
    records.close()
    with store(tmp_path) as recovered:
        assert recovered.execution("a") == receipt
        state = OperationState()
        report = fail_interrupted_requests(recovered, state)
        assert (
            report.failed_requests,
            report.uncertain_executions,
            report.confirmed_executions,
        ) == (1, 0, 1)
        failure = recovered.get("a")[1].failure
        assert failure.execution_effect == "completed"
        assert failure.completed_updates == receipt.completed_updates
        assert failure.accumulation == receipt.accumulation
        assert failure.history_id == receipt.history_id
        assert not failure.state_valid and not state.boundary.valid
        assert fail_interrupted_requests(recovered, state).failed_requests == 0


def test_restart_keeps_confirmed_update_count_without_restoring_live_counters(tmp_path):
    command = replace(request(), operation="step", mode="training")
    receipt = ExecutionReceipt(command.identity, "step", "history", "completed", 5)
    with store(tmp_path) as records:
        records.submit(command)
        records.accept("a")
        records.dispatch("a")
        records.executed(receipt)
    with store(tmp_path) as recovered:
        state = OperationState()
        fail_interrupted_requests(recovered, state)
        assert recovered.get("a")[1].failure.completed_updates == 5
        assert state.boundary.completed_updates == 0 and not state.boundary.valid
