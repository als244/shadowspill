"""Takeover retires unsent holes and fences uploads while accepted work finishes."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from io import BytesIO
from threading import Event

import pytest

from engine_tests.server.test_request_records import store
from engine_tests.training.test_request_inputs import Prepared, queue, uploaded
from shadowspill.training.session.writers import WriterOwnership
from shadowspill_client.protocol.results import ComputeResult


def with_lease(command, lease):
    return replace(
        command,
        identity=replace(
            command.identity,
            writer_id=lease.writer_id,
            writer_epoch=lease.writer_epoch,
            server_incarnation=lease.server_incarnation,
        ),
    )


def publish(receipt, work):
    return ComputeResult(
        receipt.identity,
        receipt.operation,
        "history",
        receipt.outcome,
        receipt.completed_updates,
        receipt.accumulation,
    )


def test_takeover_waits_for_rejection_boundary_and_retires_unsent_old_holes(tmp_path):
    running, release = Event(), Event()
    with store(tmp_path) as records:
        state, dispatcher, inputs = queue(records)
        writers = WriterOwnership(records, dispatcher.order, inputs)
        a, a_bytes = uploaded("a", 1)
        b, _ = uploaded("b", 3)  # Old sequence 2 never reached the server.
        inputs.register(a, staging_bytes=2048)
        inputs.receive("a", BytesIO(a_bytes))
        inputs.register(b, staging_bytes=2048)

        def execute(transition):
            inputs.take("a")
            running.set()
            assert release.wait(timeout=3)
            return Prepared(None)

        with ThreadPoolExecutor(max_workers=1) as threads:
            pending = threads.submit(dispatcher.run, "a", execute, publish)
            assert running.wait(timeout=3)
            attachment = writers.attach("new", "server")
            assert not attachment.ready and attachment.lease.next_sequence == 4
            assert writers.attach("new", "server") == attachment
            c, c_bytes = uploaded("c", 4)
            c = with_lease(c, attachment.lease)
            with pytest.raises(RuntimeError, match="attachment is waiting"):
                inputs.register(c, staging_bytes=2048)
            assert records.get("c") is None
            assert inputs.register(a, staging_bytes=2048).state == "running"
            release.set()
            assert pending.result(timeout=3).state == "completed"
        inputs.finish("a")
        assert records.get("b")[1].failure.code == "writer_fenced"
        assert records.at_sequence(2) is None
        assert writers.status("new").ready
        assert state.boundary.valid and inputs.staging.used == (0, 0)
        inputs.register(c, staging_bytes=2048)
        assert inputs.receive("c", BytesIO(c_bytes)).state == "accepted"
        assert records.get("c")[1].failure is None


def test_accepted_work_survives_immediate_takeover_and_new_work_keeps_order(tmp_path):
    with store(tmp_path) as records:
        state, dispatcher, inputs = queue(records)
        writers = WriterOwnership(records, dispatcher.order, inputs)
        a, a_bytes = uploaded("a", 1)
        inputs.register(a, staging_bytes=2048)
        inputs.receive("a", BytesIO(a_bytes))
        attachment = writers.attach("new", "server")
        assert attachment.ready
        b, b_bytes = uploaded("b", 2)
        with pytest.raises(PermissionError, match="fenced"):
            inputs.register(b, staging_bytes=2048)
        b = with_lease(b, attachment.lease)
        inputs.register(b, staging_bytes=2048)
        assert inputs.receive("b", BytesIO(b_bytes)).state == "accepted"
        assert records.get("a")[1].state == "accepted"
        for key in ("a", "b"):
            inputs.take(key)
            assert (
                dispatcher.run(key, lambda t: Prepared(None), publish).state
                == "completed"
            )
            inputs.finish(key)
        assert state.boundary.valid
        with pytest.raises(PermissionError, match="fenced"):
            writers.status("writer")
        with pytest.raises(PermissionError, match="retired"):
            writers.attach("writer", "server")
        assert state.boundary.valid and writers.status("new").ready


def test_rejected_active_upload_keeps_storage_after_new_writer_is_ready(tmp_path):
    entered, release = Event(), Event()

    class SlowReader(BytesIO):
        def readinto(self, buffer):
            entered.set()
            assert release.wait(timeout=3)
            return super().readinto(buffer)

    with store(tmp_path) as records:
        state, dispatcher, inputs = queue(records, size=2048)
        writers = WriterOwnership(records, dispatcher.order, inputs)
        a, a_bytes = uploaded()
        inputs.register(a, staging_bytes=2048)
        with ThreadPoolExecutor(max_workers=1) as threads:
            pending = threads.submit(inputs.receive, "a", SlowReader(a_bytes))
            assert entered.wait(timeout=3)
            attached = writers.attach("new", "server")
            assert attached.ready and inputs.staging.used == (2048, 1)
            b, b_bytes = uploaded("b", 2)
            inputs.register(with_lease(b, attached.lease), staging_bytes=2048)
            with pytest.raises(TimeoutError):
                inputs.receive("b", BytesIO(b_bytes))
            release.set()
            assert pending.result(timeout=3).failure.code == "writer_fenced"
        assert inputs.receive("b", BytesIO(b_bytes)).state == "accepted"
        assert state.boundary.valid


def test_retrying_current_attachment_does_not_reject_its_own_pending_upload(tmp_path):
    with store(tmp_path) as records:
        state, dispatcher, inputs = queue(records)
        writers = WriterOwnership(records, dispatcher.order, inputs)
        a, data = uploaded()
        inputs.register(a, staging_bytes=2048)
        original = writers.attach("writer", "server")
        assert original.ready and original.lease.writer_epoch == 1
        assert writers.attach("writer", "server") == original
        assert inputs.receive("a", BytesIO(data)).state == "accepted"
        assert state.boundary.valid


def test_writer_commit_failure_stops_live_state_and_recovers_original_transfer(
    tmp_path, monkeypatch
):
    records = store(tmp_path)
    state, dispatcher, inputs = queue(records)
    writers = WriterOwnership(records, dispatcher.order, inputs)
    command, data = uploaded()
    inputs.register(command, staging_bytes=2048)
    inputs.receive("a", BytesIO(data))

    def failed_index(lease):
        raise OSError("writer index failed after durable claim")

    monkeypatch.setattr(records._index, "add_writer", failed_index)
    with pytest.raises(OSError, match="durable claim"):
        writers.attach("new", "server")
    assert not state.boundary.valid
    inputs.abort()
    assert inputs.staging.used == (0, 0)
    records.close()
    with store(tmp_path) as recovered:
        assert recovered.writer.writer_id == "new"
        assert recovered.writer.writer_epoch == 2
        assert recovered.get("a")[1].state == "accepted"
