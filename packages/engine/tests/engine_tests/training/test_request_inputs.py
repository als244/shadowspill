"""Real binary uploads, owned snapshots and validation before durable acceptance."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from io import BytesIO
from threading import Event

import numpy as np
import pytest

from engine_tests.server.test_request_records import request, store
from shadowspill.training.inputs.validation import validate_batch
from shadowspill.training.session.dispatch import RequestDispatcher
from shadowspill.training.session.inputs import RequestInputs
from shadowspill.training.session.state import OperationState
from shadowspill_client import BatchConfig, Datum, Execution
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.inputs import Capacity, InputSchema, TensorField
from shadowspill_client.protocol.payload import MEDIA_TYPE, snapshot_batch
from shadowspill_client.protocol.requests import ArtifactRef
from shadowspill_client.protocol.results import ComputeResult


@dataclass
class Prepared:
    value: object
    closed: bool = False

    def close(self):
        self.value = None
        self.closed = True


SCHEMA = InputSchema(
    fields=[TensorField(("inputs", "x"), "float32", (2,))],
    capacities=[Capacity("examples", "records", "count")],
)
LIMITS = BatchConfig(limits={"examples": 2})


def prepare(request, snapshot, budget):
    assert snapshot is not None
    return Prepared(validate_batch(snapshot, SCHEMA, LIMITS))


def uploaded(name="a", sequence=1, values=None):
    batch = [
        Datum({"x": np.asarray([1, 2] if values is None else values, dtype=np.float32)})
    ]
    with snapshot_batch(batch, BufferBudget(1 << 20)) as snapshot:
        data = b"".join(snapshot.chunks())
        reference = ArtifactRef(
            name + "-input", snapshot.sha256, snapshot.size_bytes, MEDIA_TYPE
        )
    return replace(request(name, sequence), payload=reference, datum_count=1), data


def queue(records, *, count=3, size=8192, callback=prepare):
    state = OperationState()
    dispatcher = RequestDispatcher(records, state, max_pending=count)
    inputs = RequestInputs(
        records,
        dispatcher.order,
        Execution(max_pending_requests=count, max_staging_bytes=size),
        callback,
    )
    return state, dispatcher, inputs


def test_out_of_order_real_uploads_cannot_overtake_and_finish_releases_ownership(
    tmp_path,
):
    with store(tmp_path) as records:
        state, dispatcher, inputs = queue(records)
        a, a_bytes = uploaded("a", 1)
        b, b_bytes = uploaded("b", 2)
        inputs.register(b, staging_bytes=2048)
        with pytest.raises(TimeoutError):
            inputs.receive("b", BytesIO(b_bytes))
        inputs.register(a, staging_bytes=2048)
        assert inputs.receive("b", BytesIO(b_bytes)).state == "uploading"
        assert inputs.receive("a", BytesIO(a_bytes)).state == "accepted"
        assert records.get("b")[1].state == "accepted"
        assert inputs.staging.used == (4096, 2)
        for command in (a, b):
            key = command.identity.request_id
            prepared = inputs.take(key)
            assert prepared.value.owner.datums[0].inputs["x"].array().tolist() == [1, 2]

            def execute(transition):
                return Prepared(None)

            def publish(receipt, work):
                return ComputeResult(
                    receipt.identity,
                    receipt.operation,
                    "history",
                    receipt.outcome,
                    receipt.completed_updates,
                    receipt.accumulation,
                )

            dispatcher.run(key, execute, publish)
            inputs.finish(key)
            assert prepared.closed
        assert inputs.staging.used == (0, 0) and state.boundary.valid


@pytest.mark.parametrize("failure", ["checksum", "shape", "count", "truncated"])
def test_bad_complete_upload_fails_queued_suffix_and_releases_idle_owners(
    tmp_path, failure
):
    with store(tmp_path) as records:
        state, _, inputs = queue(records)
        a, data = uploaded(values=[1, 2, 3] if failure == "shape" else None)
        if failure == "checksum":
            a = replace(a, payload=replace(a.payload, sha256="0" * 64))
        elif failure == "count":
            a = replace(a, datum_count=2)
        elif failure == "truncated":
            data = data[:-1]
        b, b_data = uploaded("b", 2)
        inputs.register(a, staging_bytes=2048)
        inputs.register(b, staging_bytes=2048)
        assert inputs.receive("b", BytesIO(b_data)).state == "uploading"
        failed = inputs.receive("a", BytesIO(data))
        assert failed.failure.execution_effect == "not_started"
        assert failed.failure.code == "input_failed" and failed.acceptance is None
        assert records.get("b")[1].failure.failed_dependencies == ("a",)
        assert inputs.staging.used == (0, 0) and state.boundary.valid


def test_registration_failure_does_not_burn_sequence_or_persist_metadata(
    tmp_path,
):
    with store(tmp_path) as records:
        _, _, inputs = queue(records, count=1)
        a, data = uploaded()
        stale = replace(a, identity=replace(a.identity, writer_epoch=9))
        with pytest.raises(PermissionError, match="fenced"):
            inputs.register(stale, staging_bytes=2048)
        assert records.last_sequence == 0 and inputs.staging.used == (0, 0)
        inputs.register(a, staging_bytes=2048)
        b, _ = uploaded("b", 2)
        with pytest.raises(TimeoutError, match="window is full"):
            inputs.register(b, staging_bytes=2048)
        assert records.get("b") is None
        assert inputs.receive("a", BytesIO(data)).state == "accepted"


def test_identical_retry_never_decodes_or_preprocesses_a_second_copy(tmp_path):
    calls = []

    def inspect(request, snapshot, budget):
        calls.append(request.identity.request_id)
        return prepare(request, snapshot, budget)

    with store(tmp_path) as records:
        _, _, inputs = queue(records, callback=inspect)
        a, data = uploaded()
        inputs.register(a, staging_bytes=2048)
        accepted = inputs.receive("a", BytesIO(data))
        assert inputs.register(a, staging_bytes=2048) == accepted
        assert inputs.receive("a", BytesIO(b"malformed retry")) == accepted
        assert calls == ["a"] and inputs.staging.used == (2048, 1)
        with pytest.raises(ValueError, match="different content"):
            inputs.register(replace(a, datum_count=2), staging_bytes=2048)


def test_fencing_active_reader_preserves_storage_until_that_reader_returns(tmp_path):
    entered, release = Event(), Event()

    class SlowReader(BytesIO):
        def readinto(self, buffer):
            entered.set()
            assert release.wait(timeout=3)
            return super().readinto(buffer)

    with store(tmp_path) as records:
        _, _, inputs = queue(records, size=2048)
        a, data = uploaded()
        inputs.register(a, staging_bytes=2048)
        with ThreadPoolExecutor(max_workers=1) as threads:
            pending = threads.submit(inputs.receive, "a", SlowReader(data))
            assert entered.wait(timeout=3)
            records.claim_writer("new", "server")
            failed = inputs.reject(
                "a", code="writer_fenced", message="Writer was replaced."
            )
            assert failed.failure.code == "writer_fenced"
            assert inputs.staging.used == (2048, 1) and not pending.done()
            release.set()
            assert pending.result(timeout=3) == failed
        assert inputs.staging.used == (0, 0)


def test_rejection_wakes_waiting_upload_without_holding_the_ordering_lock(tmp_path):
    waiting = Event()
    with store(tmp_path) as records:
        _, _, inputs = queue(records, size=2048)
        a, _ = uploaded()
        b, b_data = uploaded("b", 2)
        inputs.register(a, staging_bytes=2048)
        inputs.register(b, staging_bytes=2048)

        def upload_later():
            waiting.set()
            return inputs.receive("b", BytesIO(b_data), timeout=3)

        with ThreadPoolExecutor(max_workers=1) as threads:
            pending = threads.submit(upload_later)
            assert waiting.wait(timeout=3)
            failed = inputs.reject("a", code="expired", message="Upload expired.")
            assert failed.failure.code == "expired"
            assert pending.result(timeout=3).failure.failed_dependencies == ("a",)
        assert inputs.staging.used == (0, 0)


@pytest.mark.parametrize("boundary", ["submit", "accept"])
def test_durable_failure_stops_live_state_and_releases_completed_producer(
    tmp_path, monkeypatch, boundary
):
    records = store(tmp_path)
    state, _, inputs = queue(records)
    command, data = uploaded()

    def unavailable(*args):
        raise OSError("query index failed after durable commit")

    if boundary == "submit":
        monkeypatch.setattr(records._index, "insert", unavailable)
        with pytest.raises(OSError, match="durable commit"):
            inputs.register(command, staging_bytes=2048)
    else:
        inputs.register(command, staging_bytes=2048)
        monkeypatch.setattr(records._index, "update", unavailable)
        with pytest.raises(RuntimeError, match="reconcile"):
            inputs.receive("a", BytesIO(data))
    assert not state.boundary.valid and inputs.staging.used == (0, 0)
    records.close()
    with store(tmp_path) as recovered:
        assert recovered.get("a")[1].state == (
            "uploading" if boundary == "submit" else "accepted"
        )


def test_tensor_free_commands_use_the_same_order_and_complete_ownership(tmp_path):
    def no_inputs(request, snapshot, budget):
        assert snapshot is None and request.operation == "optim_step"
        return Prepared(None)

    with store(tmp_path) as records:
        _, _, inputs = queue(records, callback=no_inputs)
        command = replace(request(), operation="optim_step", mode=None)
        inputs.register(command, staging_bytes=1)
        accepted = inputs.receive("a")
        assert accepted.state == "accepted" and inputs.staging.used == (1, 1)
        owned = inputs.take("a")
        with pytest.raises(RuntimeError, match="unavailable"):
            inputs.take("a")
        inputs.finish("a")
        assert owned.closed and inputs.staging.used == (0, 0)


def test_fatal_abort_releases_idle_inputs_without_querying_unavailable_records(
    tmp_path,
):
    records = store(tmp_path)
    state, _, inputs = queue(records)
    a, a_data = uploaded("a", 1)
    b, b_data = uploaded("b", 2)
    for command, data in ((a, a_data), (b, b_data)):
        inputs.register(command, staging_bytes=2048)
        inputs.receive(command.identity.request_id, BytesIO(data))
    owned = inputs.take("a")
    records.close()
    inputs.abort()
    assert not state.boundary.valid and inputs.staging.used == (2048, 1)
    assert not owned.closed
    # The execution owner releases only when its consumers have finished.
    inputs.finish("a")
    assert owned.closed and inputs.staging.used == (0, 0)
