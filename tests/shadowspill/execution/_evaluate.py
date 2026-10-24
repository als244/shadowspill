"""Cold worker alternates separately built Plans without model or compiler access."""

import importlib.abc
import sys
from pathlib import Path

import pytest
import torch
import torch._inductor.compile_fx as compiler
import triton
from torch.utils._pytree import tree_map


class BlockPreparation(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(
            fullname == name or fullname.startswith(name + ".")
            for name in (
                "shadowspill.build",
                "shadowspill.profiling.measure",
                "shadowspill.store",
                "shadowspill.training._legacy_trainer",
                "shadowspill.training.backends",
                "shadowspill.pytorch.capture",
                "shadowspill.pytorch.compilation",
            )
        ):
            raise AssertionError(f"execution imported {fullname}")


def forbidden(*args, **kwargs):
    raise AssertionError("execution tried to compile or calibrate")


def compilation_finished(**event):
    assert event["cache_hit"], f"unexpected compilation: {event['src'].name}"


compiler.compile_fx = compiler.compile_fx_inner = forbidden
triton.knobs.compilation.listener = compilation_finished
sys.meta_path.insert(0, BlockPreparation())
from shadowspill import Executor, Plan, Trainer, load_checkpoint, save_checkpoint  # noqa: E402
from shadowspill.planning.shared import shared_memory  # noqa: E402
from shadowspill.runtime import Runtime  # noqa: E402

Runtime._calibrate = forbidden
directory, mode = Path(sys.argv[1]), sys.argv[2]
plan, evaluation = (
    Plan.load(directory / f"{name}.ssp") for name in ("training", "evaluation")
)
oracle = torch.load(directory / "oracle.pt", weights_only=True)


def fill(request, chunk):
    assert chunk.nbytes <= 40 and chunk.device.type == "cpu"
    values = oracle["initial"][request.name].reshape(-1)
    chunk.copy_(values[request.offset : request.offset + chunk.numel()])


def snapshot(executor):
    values = {}

    def collect(request, chunk):
        values.setdefault(
            request.name, torch.empty(request.storage_elements, dtype=request.dtype)
        )
        values[request.name][request.offset : request.offset + chunk.numel()].copy_(
            chunk
        )

    executor.export_tensors(collect, chunk_bytes=40)
    return values


def cpu(value):
    return tree_map(lambda t: t.cpu() if isinstance(t, torch.Tensor) else t, value)


def check_output(actual, expected):
    # Inductor fuses the BF16 prediction epilogue; the FP32 scalar loss inherits
    # that rounding difference. Keep the original tolerances for other outputs.
    for name, value in cpu(actual).items():
        tolerance = dict(rtol=4e-3, atol=0.03) if name == "loss" else {}
        torch.testing.assert_close(value, expected[name], **tolerance)


trainer = Trainer(plan, evaluation_plan=evaluation) if mode == "trainer" else None
executor = (
    trainer.executor if trainer else Executor(plan, plans={"evaluation": evaluation})
)
executor.initialize(fill, chunk_bytes=40)
joint = shared_memory(plan, {"evaluation": evaluation})
assert (
    sum(executor.runtime._installed.admitted_layout_bytes.values())
    == joint.fixed_slice_bytes
)
primary, additional = (
    executor._paths[0].storage,
    executor._named_paths["evaluation"].storage,
)
for local, shared in joint.aliases["evaluation"].items():
    assert primary.bridge.objects.runtime_object_id(
        shared
    ) == additional.bridge.objects.runtime_object_id(local)
with pytest.raises(ValueError, match="unknown admitted Plan"):
    executor.run(plan="missing")
with pytest.raises(ValueError, match="gradient-accumulation"):
    executor.run(plan="evaluation", accumulate_gradients=True)

previous = None
for index in range(4):
    before = snapshot(executor)
    inputs = dict(
        args=[(oracle["evaluation"],)], kwargs=[oracle["evaluation_kwargs"]]
    )
    if trainer:
        actual = trainer.evaluate(**inputs)
        assert trainer.step_count == index
        assert all(t.device.type == "cpu" for t in actual.values())
    else:
        pending = executor.run(plan="evaluation", **inputs)
        actual = pending.result()
    check_output(actual, oracle["expected"][index])
    torch.testing.assert_close(snapshot(executor), before, rtol=0, atol=0)
    if previous is not None:
        check_output(previous, oracle["expected"][index - 1])
    previous = actual
    if index == 3:
        break
    if trainer:
        result = trainer.step(args=oracle["examples"])
        assert result.step == index + 1
    else:
        result = executor.run(args=oracle["examples"]).result()
    torch.testing.assert_close(
        torch.tensor([float(x) for x in result.losses]),
        torch.tensor(oracle["losses"][index]),
        # Inductor can fuse the BF16 bias/scale epilogue before the FP32 sum;
        # eager rounds each pointwise operation separately.
        rtol=4e-3,
        atol=0.03,
    )
    if index == 0:
        checkpoint = directory / f"{mode}-checkpoint.ssp"
        if trainer:
            trainer.save(checkpoint, chunk_bytes=40)
        else:
            save_checkpoint(executor, checkpoint, chunk_bytes=40)
        saved = snapshot(executor)
        executor.initialize(lambda request, chunk: chunk.zero_(), restore=True)
        if trainer:
            trainer.load(checkpoint, chunk_bytes=28)
            assert trainer.step_count == 1
        else:
            load_checkpoint(executor, checkpoint, chunk_bytes=28)
        torch.testing.assert_close(snapshot(executor), saved, rtol=0, atol=0)
actual_state = snapshot(executor)
for name, expected in oracle["final"].items():
    torch.testing.assert_close(actual_state[name].reshape(expected.shape), expected)
assert not any(name.startswith("plans.evaluation.weight") for name in actual_state)
assert any(name.startswith("plans.evaluation.constant.") for name in actual_state)
del actual, previous, result
if trainer:
    trainer.close()
else:
    del pending
    executor.close()
print("Qualified shared evaluation " + mode, flush=True)
