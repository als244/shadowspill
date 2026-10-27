"""Quantized update/evaluation/checkpoint replay from saved Plans only."""

import json
import sys
import zipfile
from pathlib import Path

import torch
import triton

from tests.shadowspill.execution._cold import forbid_preparation

forbid_preparation(
    "tests.shadowspill.pytorch.representation_fixtures", "workloads.models"
)
from shadowspill import Plan, Trainer  # noqa: E402


def compilation_finished(**event):
    assert event["cache_hit"], f"unexpected compilation: {event['src'].name}"


triton.knobs.compilation.listener = compilation_finished
root, mode = Path(sys.argv[1]), sys.argv[2]
plan, evaluation = (
    Plan.load(root / f"{name}.ssp") for name in ("training", "evaluation")
)
oracle = torch.load(root / "oracle.pt", weights_only=True)
for saved in (plan, evaluation):
    empty = next(
        r for r in saved.program.bindings["registrations"] if r["name"] == "empty"
    )
    assert saved.program.bindings["model_tensor_geometry"][empty["object_id"]][
        "shape"
    ] == (0, 3)


def fill(request, chunk):
    assert chunk.device.type == "cpu" and chunk.nbytes <= 40
    values = oracle["initial"][request.name].reshape(-1)
    chunk.copy_(values[request.offset : request.offset + chunk.numel()])


def snapshot(executor):
    values = {}

    def collect(request, chunk):
        assert chunk.nbytes <= 40
        values.setdefault(
            request.name, torch.empty(request.storage_elements, dtype=request.dtype)
        )[request.offset : request.offset + chunk.numel()].copy_(chunk)

    executor.export_tensors(collect, chunk_bytes=40)
    return values


def check(trainer, index):
    result = trainer.step(args=[(oracle["sample"],)])
    assert abs(result.loss - oracle["losses"][index]) <= 1e-7 + abs(result.loss) * 2e-5
    output = trainer.evaluate(args=[(oracle["sample"],)])
    torch.testing.assert_close(output, oracle["outputs"][index], rtol=2e-5, atol=1e-6)
    actual = snapshot(trainer.executor)
    assert actual.keys() == oracle["states"][index].keys()
    for name, expected in oracle["states"][index].items():
        torch.testing.assert_close(
            actual[name].reshape(expected.shape), expected, rtol=2e-5, atol=1e-6
        )
    return actual


checkpoint = root / "checkpoint.ssp"
with Trainer(plan, evaluation_plan=evaluation) as trainer:
    if mode == "train":
        trainer.initialize(fill, chunk_bytes=40)
        for index in range(3):
            check(trainer, index)
        trainer.save(checkpoint, chunk_bytes=40)
        with zipfile.ZipFile(checkpoint) as archive:
            entries = json.loads(archive.read("manifest.json"))["tensors"]
        names = {entry["name"] for entry in entries}
        assert {"weight.payload", "weight.scale", "master.weight"} <= names
        assert "weight" not in names
        uninterrupted = check(trainer, 3)
        torch.save(uninterrupted, root / "uninterrupted.pt")
    else:
        # A fresh worker restores all physical leaves and the master without
        # an initializer, cached model, or import of the tensor wrapper type.
        uninterrupted = torch.load(root / "uninterrupted.pt", weights_only=True)
    trainer.load(checkpoint, chunk_bytes=28)
    assert trainer.step_count == 3
    torch.testing.assert_close(check(trainer, 3), uninterrupted, rtol=0, atol=0)
print("Qualified quantized saved-Plan update, evaluation and exact resume", flush=True)
