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

root = Path(sys.argv[1])
plans = {
    name: Plan.load(root / f"{name}.ssp")
    for name in ("evaluation", "backward0", "backward1", "step0", "step1")
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
    executor.initialize(fill, chunk_bytes=32)
    for index, args in enumerate(oracle["examples"]):
        executor.run(plan=f"step{index}", args=[args]).result()
        actual = snapshot(executor)
        for name, value in oracle["updated"][index].items():
            torch.testing.assert_close(actual[name].reshape(value.shape), value)
    assert len([name for name in actual if name.endswith(".step")]) == 2
    assert not any(
        name.startswith("plans.step") and ".optimizer." in name for name in actual
    )
print("Qualified mutable shared Plans", flush=True)
