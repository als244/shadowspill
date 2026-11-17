"""Cold saved-Plan execution with caller-owned progress and checkpoint metadata."""

import sys
from pathlib import Path

import torch

from engine_tests.execution._cold import forbid_preparation

forbid_preparation()
from shadowspill import Executor, Plan, load_checkpoint, save_checkpoint  # noqa: E402
from shadowspill.training.observations import StepObservations  # noqa: E402

directory, mode = Path(sys.argv[1]), sys.argv[2]
plan = Plan.load(directory / "plan.ssp")
oracle = torch.load(directory / "oracle.pt", weights_only=True)


def fill(request, chunk):
    assert chunk.nbytes <= 40 and chunk.device.type == "cpu"
    values = oracle["initial"][request.name].reshape(-1)
    chunk.copy_(values[request.offset : request.offset + chunk.numel()])


with Executor(plan) as executor:
    position = 0
    if mode == "resume":
        position = load_checkpoint(executor, directory / "part.ssp", chunk_bytes=28)[
            "position"
        ]
        assert position == 2
    else:
        executor.initialize(fill, chunk_bytes=40)
    target = 2 if mode == "part" else 6
    while position < target:
        examples = oracle["examples"][position]
        raw = executor.run(
            args=[value[0] for value in examples],
            kwargs=[value[1] for value in examples],
            hyperparams={"optimizer_group.0.lr": oracle["rates"][position]},
        ).result()
        result = StepObservations.collect(raw.losses, raw.metrics)
        executor.synchronize()
        del raw
        torch.testing.assert_close(
            torch.tensor(result.losses), torch.tensor(oracle["losses"][position])
        )
        assert all(item["mean"].device.type == "cpu" for item in result.metrics)
        position += 1
    if mode == "part":
        save_checkpoint(
            executor,
            directory / "part.ssp",
            chunk_bytes=40,
            metadata={"position": position},
        )
    else:
        values = {}

        def collect(request, chunk):
            values.setdefault(
                request.name, torch.empty(request.storage_elements, dtype=request.dtype)
            )
            values[request.name][request.offset : request.offset + chunk.numel()].copy_(
                chunk
            )

        executor.export_tensors(collect, chunk_bytes=40)
        assert values.keys() == oracle["final"].keys()
        for name, tensor in oracle["final"].items():
            torch.testing.assert_close(values[name].reshape(tensor.shape), tensor)
print("Qualified saved training " + mode, flush=True)
