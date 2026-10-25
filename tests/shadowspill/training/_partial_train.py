"""Execution-only partial checkpoint worker; frozen tensors come from a loader."""

import json
import sys
import zipfile
from pathlib import Path

import torch

from shadowspill import Plan, Trainer

directory, mode = Path(sys.argv[1]), sys.argv[2]
initial = torch.load(directory / "initial.pt", weights_only=True)
filled = set()


def fill(request, chunk):
    assert chunk.nbytes <= 20
    filled.add(request.name)
    value = initial[request.name].reshape(-1)
    chunk.copy_(value[request.offset : request.offset + chunk.numel()])


class Source:
    def __init__(self):
        self.position = 0

    def __iter__(self):
        while self.position < 3:
            self.position += 1
            yield {"args": [(torch.full((2, 3), self.position, dtype=torch.bfloat16),)]}

    def state_dict(self):
        return {"position": self.position}

    def load_state_dict(self, state):
        self.position = state["position"]


with Trainer(
    Plan.load(directory / "training.ssp"),
    evaluation_plan=Plan.load(directory / "evaluation.ssp"),
    frozen_state_id="immutable-base-v1",
    schedules={"optimizer_group.0.lr": lambda step: 0.01 / (step + 1)},
) as trainer:
    if mode == "resume":
        trainer.load(directory / "partial.ssp", fill_tensor=fill, chunk_bytes=20)
        assert trainer.step_count == 1 and filled == {"base", "evaluation_only"}
    else:
        trainer.initialize(fill, chunk_bytes=20)

    def checkpoint(owner, result):
        if result.step == 1:
            owner.evaluate(args=[(torch.ones(2, 3, dtype=torch.bfloat16),)])
            owner.save(directory / "partial.ssp", chunk_bytes=12)
            with zipfile.ZipFile(directory / "partial.ssp") as archive:
                manifest = json.loads(archive.read("manifest.json"))
            assert manifest["omitted_parameters"] == [
                "base",
                "evaluation_only",
                "tied_base",
            ]
            entries = {entry["name"]: entry for entry in manifest["tensors"]}
            assert entries["master.delta"]["dtype"] == "torch.float32"
            assert {"delta", "moving", "calls"} <= entries.keys()

    source = Source()
    trainer.fit(source, steps=3, run_dir=directory / mode, callbacks=[checkpoint])
    assert source.position == 3
    values = {}

    def collect(request, chunk):
        values.setdefault(
            request.name, torch.empty(request.storage_elements, dtype=request.dtype)
        )
        values[request.name][request.offset : request.offset + chunk.numel()].copy_(
            chunk
        )

    trainer.executor.export_tensors(collect, chunk_bytes=20)
    if mode == "whole":
        torch.save(values, directory / "expected.pt")
    else:
        torch.testing.assert_close(
            values,
            torch.load(directory / "expected.pt", weights_only=True),
            rtol=0,
            atol=0,
        )

        def records(name):
            return [
                json.loads(line)
                for line in (directory / name / "metrics.jsonl")
                .read_text()
                .splitlines()
            ]

        assert [row["train/loss"] for row in records("resume")] == [
            row["train/loss"] for row in records("whole")[1:]
        ]

assert "shadowspill.build" not in sys.modules
assert "shadowspill.profiling.measure" not in sys.modules
assert "shadowspill.training._legacy_trainer" not in sys.modules
print(f"Qualified partial Trainer {mode}", flush=True)
