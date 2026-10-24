"""Cold worker: a Trainer sees a Plan, inputs and tensor data, never a model."""

import importlib.abc
import sys
from pathlib import Path

import torch


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
            raise AssertionError(f"training worker imported {fullname}")


sys.meta_path.insert(0, BlockPreparation())
from shadowspill import Executor, Plan, Trainer  # noqa: E402

directory, mode = Path(sys.argv[1]), sys.argv[2]
plan = Plan.load(directory / "plan.ssp")
oracle = torch.load(directory / "oracle.pt", weights_only=True)


class Source:
    def __init__(self):
        self.position = 0

    def __iter__(self):
        while self.position < len(oracle["examples"]):
            examples = oracle["examples"][self.position]
            self.position += 1
            yield {"args": [v[0] for v in examples], "kwargs": [v[1] for v in examples]}

    def state_dict(self):
        return {"position": torch.tensor(self.position)}

    def load_state_dict(self, values):
        self.position = int(values["position"])


def fill(request, chunk):
    assert chunk.nbytes <= 40 and chunk.device.type == "cpu"
    values = oracle["initial"][request.name].reshape(-1)
    chunk.copy_(values[request.offset : request.offset + chunk.numel()])


def check(trainer, result):
    expected = oracle["losses"][result.step - 1]
    torch.testing.assert_close(torch.tensor(result.losses), torch.tensor(expected))
    assert all(item["mean"].device.type == "cpu" for item in result.metrics)


borrowed = Executor(plan) if mode == "resume" else None
with Trainer(
    plan,
    executor=borrowed,
    schedules={"optimizer_group.0.lr": lambda step: oracle["rates"][step]},
) as trainer:
    if mode == "resume":
        trainer.load(directory / "part.ssp", chunk_bytes=28)
        assert trainer.step_count == 2
    else:
        trainer.initialize(fill, chunk_bytes=40)
    source = Source()
    trainer.fit(
        source,
        steps=2 if mode == "part" else 6,
        callbacks=[check],
        run_dir=directory / mode,
    )
    assert source.position == trainer.step_count
    if mode == "part":
        trainer.save(directory / "part.ssp", chunk_bytes=40)
    else:
        values = {}

        def collect(request, chunk):
            values.setdefault(
                request.name, torch.empty(request.storage_elements, dtype=request.dtype)
            )
            values[request.name][request.offset : request.offset + chunk.numel()].copy_(
                chunk
            )

        trainer.executor.export_tensors(collect, chunk_bytes=40)
        assert values.keys() == oracle["final"].keys()
        for name, tensor in oracle["final"].items():
            torch.testing.assert_close(values[name].reshape(tensor.shape), tensor)
if borrowed is not None:
    borrowed.synchronize()  # The Trainer did not close caller-owned execution.
    borrowed.close()
print("Qualified Trainer " + mode, flush=True)
