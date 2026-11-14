"""Execute prepared external definitions with capture and compilation forbidden."""

import sys
from pathlib import Path

import torch
import triton

from engine_tests.execution._cold import forbid_preparation


def compilation_finished(**event):
    assert event["cache_hit"], f"unexpected compilation: {event['src'].name}"


triton.knobs.compilation.listener = compilation_finished
forbid_preparation()

from engine_tests.fixtures.definition_operations import example, records  # noqa: E402
from shadowspill import Executor  # noqa: E402
from shadowspill.execution.results import TrainingResult  # noqa: E402
from shadowspill.server.records.requests import RequestRecords, RunIdentity  # noqa: E402
from shadowspill.training.definitions import load_registry  # noqa: E402
from shadowspill.training.observations import to_host  # noqa: E402
from shadowspill.training.preparation.bundle import TrainingBundle  # noqa: E402
from shadowspill.training.preparation.job import JOB_NAME, PreparationJob  # noqa: E402
from shadowspill.training.session.dispatch import RequestDispatcher  # noqa: E402
from shadowspill.training.session.state import OperationState  # noqa: E402
from shadowspill_client.protocol.operations import OperationKey  # noqa: E402
from shadowspill_client.protocol.requests import ComputeRequest, RequestIdentity  # noqa: E402
from shadowspill_client.protocol.results import ComputeResult  # noqa: E402

root = Path(sys.argv[1])
registry = load_registry(PreparationJob.load(root / JOB_NAME).packages)
family = TrainingBundle.load(root / "training.ssp").recheck(registry)


def full_buffer(operation):
    return next(
        row.workers[0]
        for row in family.families[operation]
        if row.source.layout.candidate.schedule.buffers[0].max_datums == 4
    )


step = full_buffer(OperationKey("step", "metrics"))
backward = full_buffer(OperationKey("forward_backward", "metrics"))
evaluation = full_buffer(OperationKey("forward", "predictions", "evaluation"))
names = {
    "step": step.main,
    "backward": backward.main,
    "masked": backward.masked,
    "evaluation": evaluation.main,
    "optim": backward.optimizer,
}
plans = dict(family.plans[0])
primary_name = min(plans)
primary = plans.pop(primary_name)


def selected(name):
    identity = names[name]
    return None if identity == primary_name else identity


oracle = torch.load(root / "oracle.pt", weights_only=True)
resolved = family.inventory.layouts.setup.model
objective = family.inventory.layouts.setup.objective


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


def submit(name, operation, **options):
    command = ComputeRequest(
        RequestIdentity(name, "run", journal.last_sequence + 1, "writer", 1, "worker"),
        operation,
        **options,
    )
    journal.submit(command)
    return command


def invoke(
    name,
    operation,
    *,
    plan,
    args=None,
    hyperparams=None,
    loss_params=None,
    participates=None,
    observe=None,
    publication_error=False,
):
    """Internal worker-adapter test; input upload/preparation is qualified elsewhere."""
    controls = hyperparams or {}
    command = submit(
        name,
        operation,
        optim={"lr": controls["optimizer_group.0.lr"]} if controls else {},
        grad_scale=controls.get("grad_scale"),
        loss_params=loss_params or {},
    )
    dispatcher.order.validated(name)

    def execute(transition):
        assert journal.get(name)[1].state == "running"
        value = executor.run(
            plan=plan,
            args=args,
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
            observe(outputs.value)
        if publication_error:
            raise OSError("injected required output publication failure")
        return ComputeResult(
            command.identity,
            operation,
            "history",
            receipt.outcome,
            receipt.completed_updates,
            receipt.accumulation,
            optim=command.optim,
            loss_params=command.loss_params,
            grad_scale=command.grad_scale,
        )

    status = dispatcher.run(name, execute, publish, participates=participates)
    if publication_error:
        assert status.failure.code == "publication_failed", status
        assert status.failure.execution_effect == "completed"
        assert status.failure.completed_updates == 2 and status.failure.state_valid

        def forbidden(transition):
            raise AssertionError("completed optimizer was replayed")

        assert dispatcher.run(name, forbidden, publish) == status
    else:
        assert status.state == ("skipped" if participates is False else "completed"), (
            status
        )
    return status


def check_step(value):
    losses, metrics = value
    torch.testing.assert_close(losses[0], oracle["step_loss"])
    assert metrics[0]["valid_examples"].item() == 3


def check_predictions(value):
    ((loss, outputs),) = value
    assert loss is None
    torch.testing.assert_close(outputs["prediction"], oracle["predictions"])


def check_masked(value):
    ((loss, outputs),) = value
    assert loss.item() == 0.0 and outputs["valid_examples"].item() == 0


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
    executor.initialize(fill, chunk_bytes=32)
    with example(resolved, objective, "step") as (binding, adapted):
        invoke(
            "step",
            "step",
            plan=selected("step"),
            args=[binding.arguments(adapted, {"temperature": 2.0})],
            hyperparams={"optimizer_group.0.lr": 0.03, "grad_scale": 0.5},
            loss_params={"temperature": 2.0},
            participates=True,
            observe=check_step,
        )
    actual = snapshot(executor)
    for name, target in oracle["after_step"].items():
        torch.testing.assert_close(actual[name].reshape(target.shape), target)
    with example(resolved, objective, "forward_backward") as (binding, adapted):
        for index, overrides in enumerate(({"temperature": 0.5}, None)):
            if index == 1:
                submit("malformed", "forward_backward")
                submit(
                    "dependent_optim", "optim_step", dependencies=("part0", "malformed")
                )
                dispatcher.order.validated("dependent_optim")
                failed = dispatcher.order.reject(
                    "malformed",
                    code="invalid_input",
                    message="Injected malformed contribution.",
                )
                assert failed.failure.accumulation.contributions == ("part0",)
                assert journal.get("dependent_optim")[
                    1
                ].failure.failed_dependencies == ("malformed",)
            invoke(
                f"part{index}",
                "forward_backward",
                plan=selected("backward"),
                args=[binding.arguments(adapted, overrides)],
                loss_params=overrides,
                participates=True,
            )
            before = snapshot(executor)
            key = OperationKey("forward", "predictions", "evaluation")
            with example(resolved, objective, key, datums=records(targets=False)) as (
                forward,
                observed,
            ):
                args = forward.arguments(observed)
                invoke(
                    f"eval{index}",
                    "forward",
                    plan=selected("evaluation"),
                    args=[args],
                    observe=check_predictions,
                )
            torch.testing.assert_close(snapshot(executor), before, rtol=0, atol=0)
    for name, target in oracle["gradients"].items():
        torch.testing.assert_close(
            before[f"gradient.{name}"].reshape(target.shape), target
        )
    with example(resolved, objective, "step", datums=records(masked=True)) as (
        binding,
        adapted,
    ):
        invoke(
            "masked",
            "forward_backward",
            plan=selected("masked"),
            args=[binding.arguments(adapted)],
            participates=False,
            observe=check_masked,
        )
        torch.testing.assert_close(snapshot(executor), before, rtol=0, atol=0)
    invoke(
        "update",
        "optim_step",
        plan=selected("optim"),
        hyperparams={"optimizer_group.0.lr": 0.02, "grad_scale": -0.5},
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
