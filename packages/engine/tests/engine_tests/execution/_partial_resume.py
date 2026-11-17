"""Execution-only partial checkpoint worker; frozen tensors come from a loader."""

import json
import sys
import tarfile
from pathlib import Path

import torch

from shadowspill import Executor, Plan, load_checkpoint, save_checkpoint

directory, mode = Path(sys.argv[1]), sys.argv[2]
initial = torch.load(directory / "initial.pt", weights_only=True)
filled = set()


def fill(request, chunk):
    assert chunk.nbytes <= 20
    filled.add(request.name)
    value = initial[request.name].reshape(-1)
    chunk.copy_(value[request.offset : request.offset + chunk.numel()])


with Executor(
    Plan.load(directory / "training.ssp"),
    plans={"evaluation": Plan.load(directory / "evaluation.ssp")},
) as executor:
    position = 0
    if mode == "resume":
        metadata = load_checkpoint(
            executor,
            directory / "partial.ssp",
            fill_tensor=fill,
            chunk_bytes=20,
            frozen_state_id="immutable-base-v1",
        )
        position = metadata["position"]
        assert position == 1 and filled == {"base", "evaluation_only"}
    else:
        executor.initialize(fill, chunk_bytes=20)
    losses = []
    while position < 3:
        result = executor.run(
            args=[(torch.full((2, 3), position + 1, dtype=torch.bfloat16),)],
            hyperparams={"optimizer_group.0.lr": 0.01 / (position + 1)},
        ).result()
        losses.append(sum(float(value) for value in result.losses))
        del result
        executor.synchronize()
        position += 1
        if position == 1:
            executor.run(
                plan="evaluation", args=[(torch.ones(2, 3, dtype=torch.bfloat16),)]
            ).result()
            save_checkpoint(
                executor,
                directory / "partial.ssp",
                chunk_bytes=12,
                frozen_state_id="immutable-base-v1",
                metadata={"position": position},
            )
            with tarfile.open(directory / "partial.ssp") as archive:
                manifest = json.loads(archive.extractfile("manifest.json").read())
            assert manifest["omitted_parameters"] == [
                "base",
                "evaluation_only",
                "tied_base",
            ]
            entries = {entry["name"]: entry for entry in manifest["tensors"]}
            assert entries["master.delta"]["dtype"] == "torch.float32"
            assert {"moving", "calls"} <= entries.keys()
            assert "delta" not in entries
            assert manifest["derived_tensors"] == {"delta": "master.delta"}
    assert position == 3
    values = {}

    def collect(request, chunk):
        values.setdefault(
            request.name, torch.empty(request.storage_elements, dtype=request.dtype)
        )
        values[request.name][request.offset : request.offset + chunk.numel()].copy_(
            chunk
        )

    executor.export_tensors(collect, chunk_bytes=20)
    if mode == "whole":
        torch.save(values, directory / "expected.pt")
        (directory / "losses.json").write_text(json.dumps(losses))
    else:
        torch.testing.assert_close(
            values,
            torch.load(directory / "expected.pt", weights_only=True),
            rtol=0,
            atol=0,
        )
        assert losses == json.loads((directory / "losses.json").read_text())[1:]

assert "shadowspill.build" not in sys.modules
assert "shadowspill.profiling.measure" not in sys.modules
print(f"Qualified partial saved training {mode}", flush=True)
