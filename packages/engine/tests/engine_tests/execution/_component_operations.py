"""Real owned uploads through prepared inputs, durable dispatch and cold execution."""

import hashlib
import io
import sys
from contextlib import nullcontext
from dataclasses import replace
from itertools import count
from pathlib import Path

import torch
import triton

from engine_tests.execution._cold import forbid_preparation


def compilation_finished(**event):
    assert event["cache_hit"], f"unexpected compilation: {event['src'].name}"


triton.knobs.compilation.listener = compilation_finished
forbid_preparation()

from engine_tests.fixtures.component_operations import records  # noqa: E402
from shadowspill.artifacts.state_inventory import program_state  # noqa: E402
from shadowspill.execution.results import TrainingResult  # noqa: E402
from shadowspill.orchestration.launch.publication import (  # noqa: E402
    OutputFiles,
    require_output,
)
from shadowspill.server.records.requests import RequestRecords, RunIdentity  # noqa: E402
from shadowspill.training.components import (  # noqa: E402
    InitializationRuntime,
    load_registry,
)
from shadowspill.training.observations import to_host  # noqa: E402
from shadowspill.training.preparation.artifacts.bundle import TrainingBundle  # noqa: E402
from shadowspill.training.preparation.phases.job import (  # noqa: E402
    JOB_NAME,
    PreparationJob,
)
from shadowspill.training.session.dispatch import RequestDispatcher  # noqa: E402
from shadowspill.training.session.inputs import RequestInputs  # noqa: E402
from shadowspill.training.session.prepared import PreparedRequests  # noqa: E402
from shadowspill.training.session.state import OperationState  # noqa: E402
from shadowspill.training.worker.bindings import worker_call  # noqa: E402
from shadowspill.training.worker.channel import read_message, write_message  # noqa: E402
from shadowspill.training.worker.output_files import FileOutputs  # noqa: E402
from shadowspill.training.worker.protocol import WorkerControl, WorkerScope  # noqa: E402
from shadowspill.training.worker.runtime import PreparedWorker  # noqa: E402
from shadowspill_client.buffers import BufferBudget  # noqa: E402
from shadowspill_client.protocol.payload import MEDIA_TYPE, snapshot_batch  # noqa: E402
from shadowspill_client.protocol.payload_io import read_payload  # noqa: E402
from shadowspill_client.protocol.requests import (  # noqa: E402
    ArtifactRef,
    OperationRequest,
    RequestIdentity,
)
from shadowspill_client.protocol.results import OperationResult  # noqa: E402

root = Path(sys.argv[1])
registry = load_registry(PreparationJob.load(root / JOB_NAME).packages)
family = TrainingBundle.load(root / "training.ssp").recheck(registry)
prepared = PreparedRequests(family, cache_root=root)
with (root / "training.ssp").open("rb") as bundle_file:
    bundle_digest = hashlib.file_digest(bundle_file, "sha256").hexdigest()
startup = PreparedWorker.resolve(
    family,
    WorkerScope("run", "worker", "execution-1", 0),
    bundle=ArtifactRef("plans", bundle_digest, (root / "training.ssp").stat().st_size),
)
worker = startup.bindings
worker_invocations = count(1)
worker_files = {}
worker_outputs = []
fail_worker_publication = False
oracle = torch.load(root / "oracle.pt", weights_only=True)


def snapshot(executor):
    values = {}

    def collect(request, chunk):
        value = values.setdefault(
            request.name, torch.empty(request.storage_elements, dtype=request.dtype)
        )
        value[request.offset : request.offset + chunk.numel()].copy_(chunk)

    executor.export_tensors(collect, chunk_bytes=32)
    return values


state = OperationState()


class HostOutputs:
    def __init__(self, value, issue=None):
        self.value, self.issue = value, issue

    def close(self):
        self.value = None


def collect(call, value, budget):
    frame = startup.outputs[call.command.candidate, call.plan]
    assert budget is running.staging and budget.used == (0, 0)
    reference = FileOutputs.collect(output_sink, call, value, budget)
    logical = (
        (value.losses, value.metrics) if isinstance(value, TrainingResult) else value
    )
    host = to_host(logical)
    groups = (
        tuple(zip(*host, strict=True)) if isinstance(value, TrainingResult) else host
    )
    with (
        require_output(output_storage.root, output_sink.spec(call)).open(
            "rb"
        ) as stream,
        read_payload(
            stream,
            reference,
            BufferBudget(frame.frame.size.staging),
            limits=frame.frame.size,
        ) as decoded,
    ):
        for path in frame.frame.shapes:
            loss, values = groups[int(path[1])]
            tensor = loss if path[2] == "loss" else values[path[3]]
            leaf = decoded.datums[0].inputs
            for key in path[1:]:
                leaf = leaf[key]
            assert (
                leaf.data
                == tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            )
    if fail_worker_publication:
        raise OSError("injected worker output disk failure")
    worker_outputs.append(host)
    return reference


class ObservedOutputs(FileOutputs):
    def collect(self, call, value, budget):
        return collect(call, value, budget)


output_disk = BufferBudget(8 << 20, max_requests=100)
output_memory = BufferBudget(8 << 20, max_requests=100)
output_storage = OutputFiles(root=root, disk=output_disk, memory=output_memory)
output_sink = ObservedOutputs(startup.outputs, root=output_storage.root)


def exchange(message):
    incoming, outgoing = io.BytesIO(), io.BytesIO()
    write_message(incoming, message)
    incoming.seek(0)
    write_message(outgoing, worker_loop.handle(read_message(incoming)))
    outgoing.seek(0)
    return read_message(outgoing)


def submit(name, operation, *, datums=None, **options):
    wire, payload = None, None
    count = 0
    if operation not in ("optim_step", "discard", "close"):
        datums = records() if datums is None else datums
        with snapshot_batch(datums, BufferBudget(8 << 20)) as owned:
            wire = b"".join(owned.chunks())
            payload = ArtifactRef(
                name + ".inputs", owned.sha256, owned.size_bytes, MEDIA_TYPE
            )
            count = len(owned.datums)
    command = OperationRequest(
        RequestIdentity(name, "run", journal.last_sequence + 1, "writer", 1, "worker"),
        operation,
        payload,
        count,
        **options,
    )
    prepared.register(inputs, command)
    return command, None if wire is None else io.BytesIO(wire)


def invoke(
    name,
    operation,
    *,
    observe=None,
    publication_error=False,
    worker_publication_error=False,
    **options,
):
    command, body = submit(name, operation, **options)
    accepted = inputs.receive(name, body)
    assert accepted.state == "accepted", accepted
    data = inputs.take(name)

    def execute(transition):
        global fail_worker_publication
        assert journal.get(name)[1].state == "running"
        command = worker_call(data, transition, worker.scope, next(worker_invocations))
        worker_files.clear()
        if data.spool is not None:
            worker_files.update(
                {frame.reference.id: frame.path for frame in data.spool.frames.values()}
            )
        bound = worker.bind(command, lambda ref: worker_files[ref.id])
        spec = output_sink.spec(bound)
        with output_storage.reserve(spec) if spec else nullcontext() as owner:
            if owner:
                owner.begin()
            ready = exchange(command)
            assert ready.status == "ready", ready
            fail_worker_publication = worker_publication_error
            try:
                completed = exchange(WorkerControl(ready.ticket, "execute"))
            finally:
                fail_worker_publication = False
            assert completed.status == (
                "failed" if worker_publication_error else "completed"
            ), completed
            assert completed.effect == "completed"
            assert worker_loop.valid == (operation != "close")
            assert running.staging.used == (0, 0)
            if owner:
                owner.finished()
                if completed.result:
                    owner.verify(completed.result)
        return HostOutputs(
            worker_outputs.pop() if worker_outputs else None, completed.message
        )

    def publish(receipt, outputs):
        assert journal.execution(name) == receipt
        if outputs.issue:
            raise OSError(outputs.issue)
        if observe is not None:
            observe(outputs.value, data)
        if publication_error:
            raise OSError("injected required output publication failure")
        return OperationResult(
            command.identity,
            operation,
            receipt.history_id,
            receipt.outcome,
            receipt.completed_updates,
            receipt.accumulation,
            optim=data.controls.optim,
            loss_params=data.controls.loss_params,
            grad_scale=data.controls.grad_scale,
            state_valid=receipt.state_valid,
        )

    try:
        status = dispatcher.run(name, execute, publish, participates=data.participates)
        if publication_error or worker_publication_error:
            assert status.failure.code == "publication_failed", status
            assert status.failure.execution_effect == "completed"
            assert status.failure.completed_updates == 2 and status.failure.state_valid

            def forbidden(transition):
                raise AssertionError("completed optimizer was replayed")

            assert dispatcher.run(name, forbidden, publish) == status
        else:
            assert status.state == "completed", status
    finally:
        inputs.finish(name)
    assert inputs.staging.used == (0, 0)
    assert not list(root.glob("inputs-*"))
    return status


def check_step(value, data):
    losses, metrics = value
    torch.testing.assert_close(sum(losses), oracle["step_loss"])
    assert sum(row["valid_examples"].item() for row in metrics) == 3


def check_predictions(value, data):
    actual = torch.empty_like(oracle["predictions"][:3])
    for piece in data.candidate.source.layout.outputs.pieces(data.assignment):
        loss, outputs = value[piece.buffer]
        assert loss is None
        actual[piece.destination.datum][piece.destination_axes] = outputs["prediction"][
            piece.source_axes
        ]
    torch.testing.assert_close(actual, oracle["predictions"][:3])


def no_artifact(identity):
    raise AssertionError("random initialization opened an artifact")


with (
    output_storage,
    RequestRecords(
        root / "requests",
        identity=RunIdentity("owner", "session", "run"),
        cache_root=root,
    ) as journal,
    startup.open(
        InitializationRuntime(root, no_artifact),
        lambda ref: worker_files[ref.id],
        output_sink,
    ) as running,
):
    executor, worker_loop = running.executor, running.loop
    journal.claim_writer("writer", "worker")
    dispatcher = RequestDispatcher(journal, state, max_pending=3)
    inputs = RequestInputs(
        journal, dispatcher.order, prepared.setup.config.execution, prepared
    )
    assert running.staging.used == (0, 0)
    initial = snapshot(executor)
    for name, expected in oracle["initial"].items():
        torch.testing.assert_close(initial[name].reshape(expected.shape), expected)
    print("Qualified bounded installed initialization")
    print("Qualified admitted worker startup")
    invoke(
        "step",
        "step",
        optim={"lr": 0.03},
        grad_scale=0.5,
        loss_params={"temperature": 2.0},
        observe=check_step,
    )
    actual = snapshot(executor)
    for name, target in oracle["after_step"].items():
        torch.testing.assert_close(actual[name].reshape(target.shape), target)
    invoke("discarded-part", "forward_backward", loss_params={"temperature": 2.0})
    gradients = {}
    for path in (*executor._paths, *executor._named_paths.values()):
        for binding in program_state(path.plan.program).bindings:
            if binding.identity.role == "gradient":
                alias = binding.alias_group_id
                objects = path.bridge.objects
                gradients.setdefault(objects.runtime_object_id(alias), (objects, alias))
    assert gradients and all(
        objects.spill_location(alias) for objects, alias in gradients.values()
    )
    invoke("discard", "discard")
    assert (
        not state.boundary.accumulation.open and state.boundary.completed_updates == 1
    )
    assert all(
        objects.spill_location(alias) is None for objects, alias in gradients.values()
    )
    invoke("discard-empty", "discard")
    invoke(
        "eval-discarded",
        "forward",
        datums=records(targets=False),
        outputs="predictions",
        observe=check_predictions,
    )
    for index, overrides in enumerate(({"temperature": 0.5}, {})):
        if index == 1:
            bad = records()
            bad[2] = replace(
                bad[2], loss_inputs={**bad[2].loss_inputs, "class_id": 1.5}
            )
            _, body = submit("malformed", "forward_backward", datums=bad)
            submit("dependent_optim", "optim_step", dependencies=("part0", "malformed"))
            inputs.receive("dependent_optim")
            failed = inputs.receive("malformed", body)
            assert failed.failure.accumulation.contributions == ("part0",)
            assert journal.get("dependent_optim")[1].failure.failed_dependencies == (
                "malformed",
            )
            assert inputs.staging.used == (0, 0)
        invoke(f"part{index}", "forward_backward", loss_params=overrides)
        before = snapshot(executor)
        invoke(
            f"eval{index}",
            "forward",
            datums=records(targets=False),
            outputs="predictions",
            observe=check_predictions,
        )
        torch.testing.assert_close(snapshot(executor), before, rtol=0, atol=0)
    for name, target in oracle["gradients"].items():
        torch.testing.assert_close(
            before[f"gradient.{name}"].reshape(target.shape), target
        )
    boundary = state.boundary
    output_count = len(worker_outputs)
    _, body = submit(
        "empty-contribution", "forward_backward", datums=records(masked=True)
    )
    failed = inputs.receive("empty-contribution", body)
    assert (
        failed.state == "failed" and "no loss participation" in failed.failure.message
    )
    assert state.boundary == boundary and len(worker_outputs) == output_count
    torch.testing.assert_close(snapshot(executor), before, rtol=0, atol=0)
    invoke(
        "update",
        "optim_step",
        optim={"lr": 0.02},
        grad_scale=-0.5,
        worker_publication_error=True,
    )
    actual = snapshot(executor)
    for name, target in oracle["after_split"].items():
        torch.testing.assert_close(actual[name].reshape(target.shape), target)
    assert (
        state.boundary.completed_updates == 2 and not state.boundary.accumulation.open
    )
    invoke("close", "close")
    assert state.boundary.closed and executor._closed and not worker_loop.valid

assert output_disk.used == output_memory.used == (0, 0)
print("Qualified component-bound operations")
print("Qualified durable request dispatcher")
print("Qualified prepared binary request inputs")
print("Qualified saved worker commands")
print("Qualified persistent worker invocation loop")
print("Qualified worker publication failure with completed optimizer state")
print("Qualified bounded output frames from admitted Executor storage")
print("Qualified pre-reserved worker output files")
print("Qualified physical gradient discard and fresh accumulation without state reset")
print("Qualified ordered Executor close through the worker barrier")
