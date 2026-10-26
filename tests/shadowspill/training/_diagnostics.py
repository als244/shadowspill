"""Cold diagnostic worker: no model, build, profiling or compilation imports."""

import json
import sys
from pathlib import Path

import torch

from tests.shadowspill.execution._cold import forbid_preparation

forbid_preparation()
from shadowspill import Plan, Trainer  # noqa: E402

directory = Path(sys.argv[1])
oracle = torch.load(directory / "oracle.pt", weights_only=True)


def fill(request, chunk):
    values = oracle["initial"][request.name].reshape(-1)
    chunk.copy_(values[request.offset : request.offset + chunk.numel()])


def snapshot(executor):
    values = {}

    def collect(request, chunk):
        target = values.setdefault(
            request.name, torch.empty(request.storage_elements, dtype=request.dtype)
        )
        target[request.offset : request.offset + chunk.numel()].copy_(chunk)

    executor.export_tensors(collect, chunk_bytes=32)
    return values


with Trainer(Plan.load(directory / "plan.ssp")) as trainer:
    trainer.initialize(fill)
    trainer.save(directory / "initial.ssp")
    initial = snapshot(trainer.executor)
    counters = {
        n: v
        for n, v in initial.items()
        if n.startswith("optimizer.") and n.endswith(".step")
    }
    assert counters and all(value.dtype == torch.int64 for value in counters.values())
    path = trainer.executor._paths[0]
    assert all(path.controls.bindings[name]["object_id"] is None for name in counters)
    torch.manual_seed(1709)
    trainer.step(args=[oracle["args"]])
    first = snapshot(trainer.executor)
    trainer.load(directory / "initial.ssp")
    torch.manual_seed(1709)
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    summary = trainer.diagnose(
        args=[oracle["args"]], directory=directory / "startup", warmup=2
    )
    assert summary["buffer_snapshot_bytes"] == 8
    assert trainer.step_count == 0
    torch.testing.assert_close(initial, snapshot(trainer.executor), rtol=0, atol=0)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    assert (directory / "startup/timelines/traced.html").is_file()
    diagnostics = json.loads((directory / "startup/step.json").read_text())
    assert diagnostics["tasks"] and diagnostics["transfers"]["fetch"]
    actual = trainer.step(args=[oracle["args"]])
    torch.testing.assert_close(first, snapshot(trainer.executor), rtol=0, atol=0)
    for index, expected in enumerate(oracle["expected"]):
        if index:
            actual = trainer.step(args=[oracle["args"]])
        torch.testing.assert_close(
            torch.tensor(actual.loss), torch.tensor(expected["loss"])
        )
        values = snapshot(trainer.executor)
        for name, expected_value in expected["values"].items():
            torch.testing.assert_close(
                values[name].reshape_as(expected_value),
                expected_value,
                check_dtype=not name.endswith(".step"),
            )
    # Diagnostics must also leave already-populated moments and step counters intact.
    current = snapshot(trainer.executor)
    step = trainer.step_count
    trainer.diagnose(args=[oracle["args"]], directory=directory / "again", warmup=0)
    torch.testing.assert_close(current, snapshot(trainer.executor), rtol=0, atol=0)
    assert trainer.step_count == step
    loads = 0

    def fail_second_input(index):
        global loads
        loads += 1
        if loads == 2:
            raise ValueError("input loader failed")
        return oracle["args"], {}

    try:
        trainer.diagnose(
            input_loader=fail_second_input, directory=directory / "failure", warmup=2
        )
    except ValueError as error:
        assert "input loader failed" in str(error)
    else:
        raise AssertionError("input-loader failure was swallowed")
    torch.testing.assert_close(current, snapshot(trainer.executor), rtol=0, atol=0)
    assert trainer.step_count == step
    trainer.save(directory / "populated.ssp")
    trainer.step(args=[oracle["args"]])
    updated = snapshot(trainer.executor)
    trainer.load(directory / "populated.ssp")
    trainer.step(args=[oracle["args"]])
    torch.testing.assert_close(updated, snapshot(trainer.executor), rtol=0, atol=0)
print("Qualified saved startup diagnostics")
