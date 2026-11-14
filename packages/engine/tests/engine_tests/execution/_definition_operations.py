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

from engine_tests.fixtures.definition_operations import definitions, example, records  # noqa: E402
from engine_tests.training.test_setup_resolution import installed  # noqa: E402
from shadowspill import Executor  # noqa: E402
from shadowspill.training.observations import to_host  # noqa: E402
from shadowspill.training.preparation.bundle import TrainingBundle  # noqa: E402
from shadowspill.training.session.state import OperationState  # noqa: E402
from shadowspill_client.protocol.operations import OperationKey  # noqa: E402

root = Path(sys.argv[1])
family = TrainingBundle.load(root / "training.ssp").recheck(installed())


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
    return "" if identity == primary_name else identity


oracle = torch.load(root / "oracle.pt", weights_only=True)
resolved, objective, _ = definitions()


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
with Executor(primary, plans=plans) as executor:
    executor.initialize(fill, chunk_bytes=32)
    with example(resolved, objective, "step") as (binding, adapted):
        result = executor.run(
            plan=selected("step"),
            args=[binding.arguments(adapted, {"temperature": 2.0})],
            hyperparams={"optimizer_group.0.lr": 0.03, "grad_scale": 0.5},
        ).result()
        torch.testing.assert_close(to_host(result.losses)[0], oracle["step_loss"])
        assert to_host(result.metrics)[0]["valid_examples"].item() == 3
        del result
        executor.synchronize()
    actual = snapshot(executor)
    for name, target in oracle["after_step"].items():
        torch.testing.assert_close(actual[name].reshape(target.shape), target)
    with example(resolved, objective, "forward_backward") as (binding, adapted):
        for index, overrides in enumerate(({"temperature": 0.5}, None)):
            transition = state.begin(
                "forward_backward", f"part{index}", participates=True
            )
            executor.run(
                plan=selected("backward"),
                args=[binding.arguments(adapted, overrides)],
                accumulate_gradients=transition.accumulate_gradients,
            ).result()
            state.complete(transition)
            before = snapshot(executor)
            key = OperationKey("forward", "predictions", "evaluation")
            with example(resolved, objective, key, datums=records(targets=False)) as (
                forward,
                observed,
            ):
                args = forward.arguments(observed)
                ((loss, outputs),) = to_host(
                    executor.run(plan=selected("evaluation"), args=[args]).result()
                )
                assert loss is None
                expected = (
                    args[0]["pixels"].flatten(start_dim=1)
                    @ oracle["after_step"]["head.weight"].T
                    + oracle["after_step"]["head.bias"]
                )
                torch.testing.assert_close(outputs["prediction"], expected)
            torch.testing.assert_close(snapshot(executor), before, rtol=0, atol=0)
    for name, target in oracle["gradients"].items():
        torch.testing.assert_close(
            before[f"gradient.{name}"].reshape(target.shape), target
        )
    with example(resolved, objective, "step", datums=records(masked=True)) as (
        binding,
        adapted,
    ):
        transition = state.begin("forward_backward", "masked", participates=False)
        assert transition.skipped
        ((loss, outputs),) = to_host(
            executor.run(
                plan=selected("masked"), args=[binding.arguments(adapted)]
            ).result()
        )
        assert loss.item() == 0.0 and outputs["valid_examples"].item() == 0
        state.complete(transition)
        torch.testing.assert_close(snapshot(executor), before, rtol=0, atol=0)
    transition = state.begin("optim_step", "update")
    executor.run(
        plan=selected("optim"),
        hyperparams={"optimizer_group.0.lr": 0.02, "grad_scale": -0.5},
    ).result()
    state.complete(transition)
    actual = snapshot(executor)
    for name, target in oracle["after_split"].items():
        torch.testing.assert_close(actual[name].reshape(target.shape), target)

print("Qualified definition-bound operations")
