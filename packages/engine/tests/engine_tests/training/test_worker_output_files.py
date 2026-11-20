"""A worker can publish only the prepared output of its exact invocation."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from engine_tests.training.test_worker_loop import call
from engine_tests.training.test_worker_output_frame import description, result
from shadowspill.orchestration.launch.publication import OutputFiles
from shadowspill.training.worker.output_files import FileOutputs
from shadowspill.training.worker.output_frame import WorkerOutputFrame
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.payload_io import read_payload


@pytest.fixture
def outputs(tmp_path):
    frame = WorkerOutputFrame(*description())
    disk, memory = BufferBudget(1 << 20, 10), BufferBudget(2 << 20, 10)
    with OutputFiles(root=tmp_path, disk=disk, memory=memory) as owner:
        sink = FileOutputs({("candidate", "main"): frame}, root=owner.root)
        bound = SimpleNamespace(command=call(), plan="main")
        yield owner, sink, frame, bound
    assert disk.used == memory.used == (0, 0)


def test_exact_permit_and_worker_budget_publish_a_verified_decodable_frame(outputs):
    owner, sink, frame, bound = outputs
    spec = sink.spec(bound)
    assert spec.size_bytes == frame.frame.size.wire
    with owner.reserve(spec) as file:
        file.begin()
        sink.check(bound)
        budget = BufferBudget(frame.frame.write_bytes, max_requests=1)
        reference = sink.collect(bound, result(), budget)
        assert reference.id == spec.id and budget.used == (0, 0)
        file.finished()
        file.verify(reference)
        with (
            file.open(reference) as stream,
            read_payload(
                stream,
                reference,
                BufferBudget(frame.frame.size.staging),
                limits=frame.frame.size,
            ) as decoded,
        ):
            assert decoded.datums[0].inputs["0"]["values"]["prediction"].shape == (2, 3)


@pytest.mark.parametrize("change", ["invocation", "scope", "request", "size"])
def test_permits_cannot_be_reused_for_a_different_invocation_or_layout(outputs, change):
    owner, sink, _, bound = outputs
    spec = sink.spec(bound)
    with owner.reserve(spec):
        if change == "size":
            sink.frames = {
                ("candidate", "main"): WorkerOutputFrame(*description(count=1))
            }
        else:
            command = bound.command
            command = {
                "invocation": lambda: replace(command, invocation=2),
                "scope": lambda: replace(
                    command, scope=replace(command.scope, worker_incarnation="new")
                ),
                "request": lambda: replace(
                    command,
                    request=replace(command.request, loss_params={"temperature": 2.0}),
                ),
            }[change]()
            bound = SimpleNamespace(command=command, plan="main")
        with pytest.raises((FileNotFoundError, ValueError)):
            sink.check(bound)


def test_missing_output_permit_fails_before_collection(outputs):
    _, sink, _, bound = outputs
    with pytest.raises(FileNotFoundError):
        sink.check(bound)
    with pytest.raises(FileNotFoundError):
        sink.collect(bound, result(), BufferBudget(1 << 20))


def test_state_command_has_no_plan_output_or_storage_permit(outputs):
    _, sink, _, bound = outputs
    bound.plan = None
    assert sink.spec(bound) is None
    sink.check(bound)
    budget = BufferBudget(1)
    assert sink.collect(bound, None, budget) is None and budget.used == (0, 0)
