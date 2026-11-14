"""Run saved operation families without access to capture or compilation."""

import sys
from pathlib import Path

import torch
import triton

from engine_tests.execution._cold import forbid_preparation


def compilation_finished(**event):
    assert event["cache_hit"], f"unexpected compilation: {event['src'].name}"


triton.knobs.compilation.listener = compilation_finished
forbid_preparation()
from shadowspill import Executor, Plan  # noqa: E402
from shadowspill.planning.shared import shared_memory  # noqa: E402
from shadowspill.training.observations import to_host  # noqa: E402
from shadowspill.training.session.state import OperationState  # noqa: E402

root = Path(sys.argv[1])
plans = {
    name: Plan.load(root / f"{name}.ssp")
    for name in (
        "evaluation",
        "backward0",
        "backward1",
        "step0",
        "step1",
        "optim",
        "complete",
        "masked_outputs",
    )
}
primary = plans.pop("evaluation")
oracle = torch.load(root / "oracle.pt", weights_only=True)


def fill(request, chunk):
    assert chunk.nbytes <= 32
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


joint = shared_memory(primary, plans)
assert any(owner == "step0" for owner, _ in joint.aliases["step1"].values())
assert joint.host["step1"] and not joint.host["step0"]
assert any(owner == "backward0" for owner, _ in joint.aliases["backward1"].values())
with Executor(primary, plans=plans) as executor:
    executor.initialize(fill, chunk_bytes=32)
    paths = executor._named_paths
    for name, mapping in joint.host.items():
        for binding, (owner, source) in mapping.items():
            assert (
                paths[name].controls.host[binding] is paths[owner].controls.host[source]
            )
    for index, args in enumerate(oracle["examples"]):
        executor.run(
            plan=f"backward{index}", args=[args], accumulate_gradients=bool(index)
        ).result()
        before = snapshot(executor)
        for name, gradient in oracle["gradients"][index].items():
            torch.testing.assert_close(
                before[f"gradient.{name}"].reshape(gradient.shape), gradient
            )
        output = executor.run(args=[(oracle["examples"][0][0],)]).result()
        expected = (
            oracle["examples"][0][0] @ oracle["initial"]["weight"]
            + oracle["initial"]["bias"]
        )
        torch.testing.assert_close(output.cpu(), expected)
        torch.testing.assert_close(snapshot(executor), before, rtol=0, atol=0)
        del output
    result = executor.run(
        plan="optim",
        hyperparams={
            plans["optim"].program.bindings["gradient_scale_control"]: oracle[
                "split_scale"
            ]
        },
    ).result()
    assert not result.losses and not result.metrics
    split = snapshot(executor)
    for name, value in oracle["split"].items():
        torch.testing.assert_close(split[name].reshape(value.shape), value)
    executor.initialize(fill, chunk_bytes=32)
    executor.run(
        plan="complete",
        args=oracle["examples"],
        hyperparams={
            plans["complete"].program.bindings["gradient_scale_control"]: oracle[
                "split_scale"
            ]
        },
    ).result()
    complete = snapshot(executor)
    for name, value in oracle["split"].items():
        torch.testing.assert_close(complete[name].reshape(value.shape), value)
    executor.initialize(fill, chunk_bytes=32)
    for index, factor in enumerate(oracle["update_scales"]):
        number = index % len(oracle["examples"])
        args = oracle["examples"][number]
        name = f"step{number}"
        controls = (
            {}
            if factor is None
            else {plans[name].program.bindings["gradient_scale_control"]: factor}
        )
        executor.run(plan=name, args=[args], hyperparams=controls).result()
        actual = snapshot(executor)
        for name, value in oracle["updated"][index].items():
            torch.testing.assert_close(actual[name].reshape(value.shape), value)
    assert len([name for name in actual if name.endswith(".step")]) == 2
    assert not any(
        name.startswith("plans.step") and ".optimizer." in name for name in actual
    )

    # The operation state selects fresh/additive paths from completed validity,
    # not call number. A masked request still observes its requested outputs
    # through a prepared forward-only computation with the same objective.
    executor.initialize(fill, chunk_bytes=32)
    state = OperationState()

    def execute(
        command, request_id, name=None, args=None, participates=None, scale=None
    ):
        transition = state.begin(command, request_id, participates=participates)
        output = None
        if transition.skipped:
            name = "masked_outputs" if args is not None else None
        if name is not None:
            controls = (
                {}
                if scale is None or transition.skipped
                else {plans[name].program.bindings["gradient_scale_control"]: scale}
            )
            result = executor.run(
                plan=name,
                args=None if args is None else [args],
                accumulate_gradients=transition.accumulate_gradients,
                hyperparams=controls,
            ).result()
            state.complete(transition)
            if isinstance(result, tuple):
                output = to_host(result)
            else:
                output = to_host((result.losses, result.metrics))
            executor.synchronize()
        else:
            state.complete(transition)
        return transition, output

    def mutable_state():
        return {
            name: value
            for name, value in snapshot(executor).items()
            if not name.startswith(("gradient.", "plans."))
        }

    masked = (*oracle["examples"][0][:2], torch.zeros_like(oracle["examples"][0][2]))
    initial = mutable_state()
    empty, outputs = execute(
        "forward_backward", "masked_first", args=masked, participates=False
    )
    assert empty.skipped and not empty.accumulate_gradients
    assert outputs[0].item() == 0.0
    torch.testing.assert_close(
        outputs[1]["prediction"],
        masked[0] @ oracle["initial"]["weight"] + oracle["initial"]["bias"],
    )
    torch.testing.assert_close(mutable_state(), initial, rtol=0, atol=0)
    first, _ = execute(
        "forward_backward", "valid_first", "backward0", oracle["examples"][0], True
    )
    assert not first.accumulate_gradients
    before = snapshot(executor)
    execute("forward_backward", "masked_later", args=masked, participates=False)
    torch.testing.assert_close(snapshot(executor), before, rtol=0, atol=0)
    evaluation = state.begin("forward", "evaluation_between_parts")
    executor.run(args=[(oracle["examples"][0][0],)]).result()
    state.complete(evaluation)
    assert state.boundary.accumulation == evaluation.before.accumulation
    torch.testing.assert_close(snapshot(executor), before, rtol=0, atol=0)
    last, _ = execute(
        "forward_backward", "valid_later", "backward1", oracle["examples"][1], True
    )
    assert last.accumulate_gradients
    execute("optim_step", "update", "optim", scale=oracle["split_scale"])
    for name, value in oracle["split"].items():
        torch.testing.assert_close(snapshot(executor)[name].reshape(value.shape), value)
    assert (
        state.boundary.completed_updates == 1 and not state.boundary.accumulation.open
    )

    before = mutable_state()
    execute("step", "masked_step", args=masked, participates=False)
    execute("forward_backward", "entirely_masked", args=masked, participates=False)
    skipped, _ = execute("optim_step", "skipped_update", "optim")
    assert skipped.skipped and state.boundary.completed_updates == 1
    torch.testing.assert_close(mutable_state(), before, rtol=0, atol=0)

    # Discard closes the generation without an optimizer call; the next valid
    # contribution must overwrite stale gradient storage rather than add to it.
    execute(
        "forward_backward", "discarded_part", "backward0", oracle["examples"][0], True
    )
    discarded = snapshot(executor)
    before = mutable_state()
    execute("discard", "discard")
    torch.testing.assert_close(mutable_state(), before, rtol=0, atol=0)
    fresh, _ = execute(
        "forward_backward",
        "fresh_after_discard",
        "backward0",
        oracle["examples"][0],
        True,
    )
    assert not fresh.accumulate_gradients
    torch.testing.assert_close(snapshot(executor), discarded, rtol=0, atol=0)
    execute("discard", "final_discard")
    from engine_tests.execution._packed_shared import qualify_packed_family

    qualify_packed_family(executor, oracle, snapshot, fill)
print("Qualified mutable shared Plans", flush=True)
