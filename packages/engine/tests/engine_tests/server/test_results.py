"""Real uploads and durable outcomes with synthetic fixed-ABI worker outputs."""

import io
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from engine_tests.fixtures.definition_operations import records
from engine_tests.server.test_artifacts import store as artifact_store
from engine_tests.server.test_request_records import request, store
from engine_tests.training.test_outputs import assemble
from engine_tests.training.test_prepared_requests import queue, upload
from shadowspill.server.results import ResultPublisher
from shadowspill.training.inputs.frames import InputFrame
from shadowspill.training.result_frames import ResultFrames
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.payload_io import read_payload
from shadowspill_client.protocol.requests import OperationRequest


class Work:
    def __init__(self, call):
        self.frames, self.files, self.values = {}, {}, {}
        self.verified = self.closed = False
        if call.candidate is None:
            return
        layout = call.candidate.source.layout
        for rank in call.candidate.workers:
            buffers = tuple(
                buffer for buffer, owner in sorted(layout.bindings) if owner == rank
            )
            fields = {}
            for index, buffer in enumerate(buffers):
                binding = layout.bindings[buffer, rank]
                values = {}
                for name in binding.contract.outputs.tensors:
                    shape = binding.tensor_shapes[name]
                    value = (
                        np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
                        + buffer * 100
                    )
                    values[name] = value
                    fields["inputs", str(index), "values", name] = value
                for name in binding.contract.outputs.metrics:
                    fields["inputs", str(index), "values", name] = np.asarray(
                        2, np.int64
                    )
                if call.key.operation != "forward":
                    fields["inputs", str(index), "loss"] = np.asarray(0.5, np.float32)
                self.values[buffer, rank] = values
            frame = InputFrame(
                {path: value.shape for path, value in fields.items()},
                {path: value.dtype.name for path, value in fields.items()},
            )
            stream = io.BytesIO()
            frame.write(stream, fields, BufferBudget(4 << 20), "worker")
            self.frames[rank] = SimpleNamespace(frame=frame, buffers=buffers)
            self.files[rank] = stream.getvalue()

    def verify(self):
        assert not self.closed
        self.verified = True

    @contextmanager
    def open(self, rank):
        assert self.verified and not self.closed
        with io.BytesIO(self.files[rank]) as stream:
            yield stream

    def close(self):
        self.closed = True


def test_discard_publishes_confirmed_state_without_tensor_scratch(tmp_path):
    memory, disk = BufferBudget(8 << 20, 1), BufferBudget(1)
    with (
        store(tmp_path) as journal,
        artifact_store(tmp_path, memory=memory) as artifacts,
        ResultPublisher(
            artifacts,
            session_id="session",
            root=tmp_path,
            memory=memory,
            disk=disk,
        ) as publisher,
    ):
        prepared, state, dispatcher, inputs = queue(journal, tmp_path)
        state.complete(state.begin("forward_backward", "prior", participates=True))
        command = OperationRequest(request().identity, "discard")
        prepared.register(inputs, command)
        assert inputs.receive("a").state == "accepted"
        call = inputs.take("a")
        work = Work(call)
        status = dispatcher.run(
            "a", lambda _: work, lambda receipt, value: publisher(call, receipt, value)
        )
        assert status.state == "completed", status
        assert work.closed and work.verified and not status.result.outputs
        assert status.result.completed_updates == 0
        assert status.result.accumulation.generation == 1
        assert not status.result.accumulation.open and disk.used == (0, 0)
        inputs.finish("a")
        inputs.staging.close()
    assert memory.used == (0, 0)


@pytest.mark.parametrize("operation", ["forward", "step"])
def test_production_publisher_restores_caller_order_and_records_loss(
    tmp_path, operation
):
    # The persistent index shares this arena: index + assembly + reader + reducer.
    memory, disk = BufferBudget(16 << 20, 4), BufferBudget(16 << 20, 10)
    with (
        store(tmp_path) as journal,
        artifact_store(tmp_path, memory=memory) as artifacts,
    ):
        prepared, state, dispatcher, inputs = queue(journal, tmp_path)
        command, wire = upload(
            records(targets=operation != "forward"),
            operation=operation,
            outputs="predictions" if operation == "forward" else "metrics",
        )
        prepared.register(inputs, command)
        inputs.receive("a", io.BytesIO(wire))
        call = inputs.take("a")
        work = Work(call)
        publisher = ResultPublisher(
            artifacts,
            session_id="session",
            root=tmp_path,
            memory=memory,
            disk=disk,
        )
        expected = assemble(
            call.candidate.source.layout.outputs, call.assignment, work.values
        )

        def publish(receipt, value):
            assert journal.execution("a") == receipt
            return publisher(call, receipt, value)

        try:
            status = dispatcher.run(
                "a", lambda _: work, publish, participates=call.participates
            )
            assert status.state == "completed", status
            assert work.closed and work.verified
            if operation == "step":
                count = len(call.candidate.source.layout.candidate.schedule.buffers)
                assert status.result.loss == 0.5 * count
                assert status.result.metrics["valid_examples"] == 2 * count
                assert (
                    status.result.completed_updates
                    == state.boundary.completed_updates
                    == 1
                )
            else:
                assert status.result.loss is None
                # Owner storage is shared, but equal run/request/output names
                # in different sessions must still identify different artifacts.
                with ResultPublisher(
                    artifacts,
                    session_id="other-session",
                    root=tmp_path,
                    memory=memory,
                    disk=disk,
                ) as other:
                    duplicate = Work(call)
                    second = other(call, journal.execution("a"), duplicate)
                    duplicate.close()
                assert second.outputs.keys() == status.result.outputs.keys()
                for name, ref in second.outputs.items():
                    original = status.result.outputs[name]
                    assert ref.id != original.id and ref.sha256 == original.sha256
                    assert artifacts.get(ref.id) == ref
                    assert artifacts.get(original.id) == original
            inputs.finish("a")
            assert call.closed
            for name, reference in status.result.outputs.items():
                with (
                    artifacts.open(reference) as stream,
                    read_payload(stream, reference, memory) as decoded,
                ):
                    values = decoded.datums[0].inputs
                    for (output, datum), value in expected.items():
                        if output == name:
                            np.testing.assert_array_equal(
                                values[str(datum)].array(), value
                            )
            assert disk.used == (0, 0)
        finally:
            if not call.closed:
                inputs.finish("a")
            inputs.staging.close()
            publisher.close()
    assert memory.used == (0, 0)


def test_failed_result_cleanup_remains_owned_for_publisher_retry(tmp_path, monkeypatch):
    memory, disk = BufferBudget(16 << 20, 10), BufferBudget(16 << 20, 10)
    with (
        store(tmp_path) as journal,
        artifact_store(tmp_path, memory=memory) as artifacts,
    ):
        prepared, state, dispatcher, inputs = queue(journal, tmp_path)
        command, wire = upload(
            records(targets=False), operation="forward", outputs="predictions"
        )
        prepared.register(inputs, command)
        inputs.receive("a", io.BytesIO(wire))
        call, close = inputs.take("a"), ResultFrames.close
        work = Work(call)
        publisher = ResultPublisher(
            artifacts,
            session_id="session",
            root=tmp_path,
            memory=memory,
            disk=disk,
        )

        def fail(_):
            raise OSError("result scratch cleanup failed")

        monkeypatch.setattr(ResultFrames, "close", fail)
        status = dispatcher.run(
            "a", lambda _: work, lambda receipt, value: publisher(call, receipt, value)
        )
        assert (
            status.failure.execution_effect == "completed"
            and status.failure.state_valid
        )
        assert state.boundary.valid and work.closed
        assert publisher._owners and disk.used[0] > 0
        assert all(owner.assignment is None for owner in publisher._owners)
        with pytest.raises(OSError):
            publisher.close()
        monkeypatch.setattr(ResultFrames, "close", close)
        publisher.close()
        assert disk.used == (0, 0)
        inputs.finish("a")
        inputs.staging.close()
    assert memory.used == (0, 0)
