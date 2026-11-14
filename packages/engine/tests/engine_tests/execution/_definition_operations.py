"""Real owned uploads through prepared inputs, durable dispatch and cold execution."""

import io
import sys
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import torch
import triton

from engine_tests.execution._cold import forbid_preparation


def compilation_finished(**event):
    assert event["cache_hit"], f"unexpected compilation: {event['src'].name}"


triton.knobs.compilation.listener = compilation_finished
forbid_preparation()

from engine_tests.fixtures.definition_operations import records  # noqa: E402
from shadowspill import Executor  # noqa: E402
from shadowspill.execution.results import TrainingResult  # noqa: E402
from shadowspill.server.records.requests import RequestRecords, RunIdentity  # noqa: E402
from shadowspill.training.definitions import load_registry  # noqa: E402
from shadowspill.training.observations import to_host  # noqa: E402
from shadowspill.training.preparation.bundle import TrainingBundle  # noqa: E402
from shadowspill.training.preparation.job import JOB_NAME, PreparationJob  # noqa: E402
from shadowspill.training.session.dispatch import RequestDispatcher  # noqa: E402
from shadowspill.training.session.inputs import RequestInputs  # noqa: E402
from shadowspill.training.session.prepared import PreparedRequests  # noqa: E402
from shadowspill.training.session.state import OperationState  # noqa: E402
from shadowspill.training.session.worker_inputs import ReadyInputLoader  # noqa: E402
from shadowspill_client.buffers import BufferBudget  # noqa: E402
from shadowspill_client.protocol.payload import MEDIA_TYPE, snapshot_batch  # noqa: E402
from shadowspill_client.protocol.requests import (  # noqa: E402
    ArtifactRef,
    ComputeRequest,
    RequestIdentity,
)
from shadowspill_client.protocol.results import ComputeResult  # noqa: E402

root = Path(sys.argv[1])
registry = load_registry(PreparationJob.load(root / JOB_NAME).packages)
family = TrainingBundle.load(root / "training.ssp").recheck(registry)
prepared = PreparedRequests(family, cache_root=root)
plans = dict(family.plans[0])
primary_name = min(plans)
primary = plans.pop(primary_name)
oracle = torch.load(root / "oracle.pt", weights_only=True)


def fill(request, chunk):
    value = oracle["initial"][request.name].reshape(-1)
    chunk.copy_(value[request.offset : request.offset + chunk.numel()])


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
    def __init__(self, value):
        self.value = value

    def close(self):
        self.value = None


def submit(name, operation, *, datums=None, **options):
    wire, payload = None, None
    count = 0
    if operation != "optim_step":
        datums = records() if datums is None else datums
        with snapshot_batch(datums, BufferBudget(8 << 20)) as owned:
            wire = b"".join(owned.chunks())
            payload = ArtifactRef(
                name + ".inputs", owned.sha256, owned.size_bytes, MEDIA_TYPE
            )
            count = len(owned.datums)
    command = ComputeRequest(
        RequestIdentity(name, "run", journal.last_sequence + 1, "writer", 1, "worker"),
        operation,
        payload,
        count,
        **options,
    )
    prepared.register(inputs, command)
    return command, None if wire is None else io.BytesIO(wire)


def invoke(name, operation, *, observe=None, publication_error=False, **options):
    command, body = submit(name, operation, **options)
    accepted = inputs.receive(name, body)
    assert accepted.state == "accepted", accepted
    data = inputs.take(name)

    def execute(transition):
        assert journal.get(name)[1].state == "running"
        identity = data.worker_plans(transition)[0]
        if identity is None:
            return HostOutputs(None)
        plan = family.plans[0][identity]
        hyperparams = {}
        if operation in ("step", "optim_step") and not transition.skipped:
            optimizer = prepared.setup.optimizer
            for control, targets in optimizer.definition.control_bindings.items():
                for target in targets:
                    hyperparams[target] = data.controls.optim[control]
            hyperparams[plan.program.bindings["gradient_scale_control"]] = (
                data.controls.grad_scale
            )
        with (
            ReadyInputLoader(data, 0) if data.spool is not None else nullcontext(None)
        ) as loader:
            value = executor.run(
                plan=None if identity == primary_name else identity,
                input_loader=loader,
                hyperparams=hyperparams,
                accumulate_gradients=transition.accumulate_gradients,
            ).result()
            owned = to_host(
                (value.losses, value.metrics)
                if isinstance(value, TrainingResult)
                else value
            )
            del value
            executor.synchronize()
        return HostOutputs(owned)

    def publish(receipt, outputs):
        assert journal.execution(name) == receipt
        if observe is not None:
            observe(outputs.value, data)
        if publication_error:
            raise OSError("injected required output publication failure")
        return ComputeResult(
            command.identity,
            operation,
            "history",
            receipt.outcome,
            receipt.completed_updates,
            receipt.accumulation,
            optim=data.controls.optim,
            loss_params=data.controls.loss_params,
            grad_scale=data.controls.grad_scale,
        )

    try:
        status = dispatcher.run(name, execute, publish, participates=data.participates)
        if publication_error:
            assert status.failure.code == "publication_failed", status
            assert status.failure.execution_effect == "completed"
            assert status.failure.completed_updates == 2 and status.failure.state_valid

            def forbidden(transition):
                raise AssertionError("completed optimizer was replayed")

            assert dispatcher.run(name, forbidden, publish) == status
        else:
            assert status.state == (
                "skipped" if data.participates is False else "completed"
            ), status
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


def check_masked(value, data):
    assert sum(loss.item() for loss, outputs in value) == 0.0
    assert sum(outputs["valid_examples"].item() for loss, outputs in value) == 0


with (
    RequestRecords(
        root / "requests",
        identity=RunIdentity("owner", "session", "run"),
        cache_root=root,
    ) as journal,
    Executor(primary, plans=plans) as executor,
):
    journal.claim_writer("writer", "worker")
    dispatcher = RequestDispatcher(journal, state, max_pending=3)
    inputs = RequestInputs(
        journal, dispatcher.order, prepared.setup.config.execution, prepared
    )
    executor.initialize(fill, chunk_bytes=32)
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
    invoke(
        "masked", "forward_backward", datums=records(masked=True), observe=check_masked
    )
    torch.testing.assert_close(snapshot(executor), before, rtol=0, atol=0)
    invoke(
        "update",
        "optim_step",
        optim={"lr": 0.02},
        grad_scale=-0.5,
        publication_error=True,
    )
    actual = snapshot(executor)
    for name, target in oracle["after_split"].items():
        torch.testing.assert_close(actual[name].reshape(target.shape), target)
    assert (
        state.boundary.completed_updates == 2 and not state.boundary.accumulation.open
    )

print("Qualified definition-bound operations")
print("Qualified durable request dispatcher")
print("Qualified prepared binary request inputs")
