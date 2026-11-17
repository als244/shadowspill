"""Real uploads and durable outcomes with synthetic fixed-ABI worker outputs."""

import hashlib
import io
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from engine_tests.fixtures.definition_operations import definitions, records
from engine_tests.server.test_artifacts import store as artifact_store
from engine_tests.server.test_http import ROOT, TOKEN, endpoint
from engine_tests.server.test_request_records import request, store
from engine_tests.server.test_service_client import address
from engine_tests.training.test_outputs import assemble
from engine_tests.training.test_prepared_requests import queue, upload
from engine_tests.training.test_setup_resolution import arguments, installed
from shadowspill.server.results import ResultPublisher
from shadowspill.training.inputs.frames import InputFrame
from shadowspill.training.preparation.capture.templates import capture_inventory
from shadowspill.training.preparation.configuration.layouts import prepare_layouts
from shadowspill.training.preparation.setup import resolve_training
from shadowspill.training.result_frames import ResultFrames
from shadowspill_client import ServiceClient
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.definitions import TensorOutput
from shadowspill_client.protocol.payload_io import read_payload
from shadowspill_client.protocol.requests import ArtifactRef, OperationRequest
from shadowspill_client.protocol.results import ArtifactIssue


class Work:
    def __init__(self, call):
        self.frames, self.files, self.values = {}, {}, {}
        self.verified = self.closed = False
        self.execution_seconds = 1.25 if call.key is not None else None
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


@pytest.mark.parametrize("failure", [None, "worker", "checksum", "publication"])
def test_optional_trace_failure_keeps_successful_update_and_durable_result(
    tmp_path, monkeypatch, failure
):
    memory, disk = BufferBudget(16 << 20, 10), BufferBudget(16 << 20, 10)
    payload = b"owned trace archive"
    reference = ArtifactRef(
        "trace", hashlib.sha256(payload).hexdigest(), len(payload), "application/zip"
    )
    with (
        store(tmp_path) as journal,
        artifact_store(tmp_path, memory=memory) as artifacts,
        ResultPublisher(
            artifacts, session_id="session", root=tmp_path, memory=memory, disk=disk
        ) as publisher,
    ):
        prepared, state, dispatcher, inputs = queue(journal, tmp_path)
        command, wire = upload(outputs="metrics")
        command = replace(command, runtime_trace=True)
        prepared.register(inputs, command)
        inputs.receive("a", io.BytesIO(wire))
        call = inputs.take("a")
        work, executed = Work(call), []

        def outcomes():
            assert work.verified and journal.execution("a") is not None
            return {
                0: ArtifactIssue(
                    "runtime_trace_failed", "optional worker failure", reference.id
                )
                if failure == "worker"
                else reference
            }

        @contextmanager
        def open_trace(rank, ref):
            assert not work.closed and rank == 0 and ref == reference
            with io.BytesIO(
                b"x" * len(payload) if failure == "checksum" else payload
            ) as stream:
                yield stream

        work.traces, work.open_trace = outcomes, open_trace
        if failure == "publication":

            def failed(*_):
                raise OSError("trace storage unavailable")

            monkeypatch.setattr(artifacts, "put", failed)

        def execute(_):
            executed.append(1)
            return work

        def publish(receipt, value):
            return publisher(call, receipt, value)

        status = dispatcher.run("a", execute, publish, participates=True)
        assert status.state == "completed" and status.result.state_valid
        assert status.result.completed_updates == state.boundary.completed_updates == 1
        assert status.result.loss is not None
        assert status.result.execution_seconds == 1.25
        assert work.closed and executed == [1]
        if failure:
            assert not status.result.traces
            assert status.result.artifact_issues[0].artifact_id == reference.id
            assert artifacts.get(reference.id) is None
        else:
            assert (
                status.result.traces == (reference,)
                and not status.result.artifact_issues
            )
            with artifacts.open(reference) as stream:
                assert stream.read() == payload
        # Response recovery returns the durable result without another update or
        # another attempt to publish the optional artifact.
        assert dispatcher.run("a", execute, publish, participates=True) == status
        assert executed == [1]
        inputs.finish("a")
        inputs.staging.close()
    assert memory.used == disk.used == (0, 0)


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


@pytest.mark.parametrize("operation", ["step", "forward_backward"])
@pytest.mark.parametrize("fault", ["assembly", "storage", "cleanup", "unverified"])
def test_verified_scalars_survive_required_output_failure_and_journal_reopen(
    tmp_path, monkeypatch, operation, fault
):
    # The same fixed-ABI CPU worker fixture declares both objective scalars and
    # a requested tensor, so failure occurs after collecting genuine frame data.
    _, objective, _ = definitions()
    definition = replace(
        objective.definition,
        outputs={
            "metrics": replace(
                objective.definition.outputs["metrics"],
                tensors={"prediction": TensorOutput("float32", (3,), "datum")},
            )
        },
    )
    inventory = capture_inventory(
        prepare_layouts(
            resolve_training(
                installed(objective=definition),
                **arguments(),
                operations={operation: ["metrics"]},
            )
        )
    )
    memory, disk = BufferBudget(16 << 20, 10), BufferBudget(16 << 20, 10)
    with (
        store(tmp_path) as journal,
        artifact_store(tmp_path, memory=memory) as artifacts,
        ResultPublisher(
            artifacts, session_id="session", root=tmp_path, memory=memory, disk=disk
        ) as publisher,
    ):
        prepared, state, dispatcher, inputs = queue(journal, tmp_path, inventory)
        command, wire = upload(operation=operation, outputs="metrics")
        prepared.register(inputs, command)
        inputs.receive("a", io.BytesIO(wire))
        call = inputs.take("a")
        work, calls = Work(call), []

        def fail(*_):
            raise OSError("injected " + fault + " failure")

        def execute(_):
            calls.append("a")
            return work

        with monkeypatch.context() as patch:
            if fault == "assembly":
                patch.setattr(ResultFrames, "assemble", fail)
            elif fault == "storage":
                patch.setattr(artifacts, "produce", fail)
            elif fault == "cleanup":
                patch.setattr(ResultFrames, "close", fail)
            else:
                patch.setattr(work, "verify", fail)
            status = dispatcher.run(
                "a",
                execute,
                lambda receipt, value: publisher(call, receipt, value),
                participates=True,
            )
        failure = status.failure
        assert (
            failure.code == "publication_failed"
            and failure.execution_effect == "completed"
        )
        assert failure.state_valid and work.closed and calls == ["a"]
        count = len(call.candidate.source.layout.candidate.schedule.buffers)
        if fault == "unverified":
            assert failure.loss is None and not failure.metrics
        else:
            assert failure.loss == 0.5 * count
            assert failure.metrics == {"valid_examples": 2 * count}
        assert failure.completed_updates == (1 if operation == "step" else 0)
        assert failure.completed_updates == state.boundary.completed_updates
        assert journal.metrics().records[0].loss == failure.loss
        assert journal.metrics().records[0].metrics == failure.metrics
        assert [row.kind for row in journal._journal.records()][-3:] == [
            "request.status",
            "request.executed",
            "request.status",
        ]
        assert (
            dispatcher.run("a", execute, lambda *_: pytest.fail("replayed")) == status
        )
        assert calls == ["a"]
        inputs.finish("a")
        inputs.staging.close()
    assert memory.used == disk.used == (0, 0)
    with store(tmp_path) as reopened:
        assert reopened.get("a")[1] == status
        assert reopened.metrics().records[0].metrics == failure.metrics
        with (
            endpoint(
                tmp_path, run=SimpleNamespace(journal=reopened, controller=None)
            ) as http,
            ServiceClient(
                address(http.client), token=TOKEN, session_id="session"
            ) as client,
        ):
            row = client.get_metrics("run").result(5).records[0]
            assert row.loss == failure.loss and row.metrics == failure.metrics
            assert http.client.request("GET", ROOT + "/requests/a").value == status
        assert client._memory.used == client._work.pending.used == (0, 0)
