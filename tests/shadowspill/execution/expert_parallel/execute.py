"""Fresh saved-Plan EP worker; model construction and preparation are forbidden."""

import json
import sys
from pathlib import Path

import torch

from tests.shadowspill.execution._cold import forbid_preparation

forbid_preparation()
from benchmarking.quickstart.phases.common import experiment, read  # noqa: E402
from benchmarking.quickstart.phases.distributed import workers  # noqa: E402
from shadowspill import Executor, Plan, load_checkpoint, save_checkpoint  # noqa: E402
from tests.shadowspill.execution._trace_checks import check_trace  # noqa: E402


def compare(executor, expected):
    values = {}
    missing = set()

    def collect(request, chunk):
        assert chunk.device.type == "cpu" and chunk.nbytes <= 8192
        if request.name not in expected:
            missing.add(request.name)
            return
        wanted = expected[request.name].reshape(-1)
        target = values.setdefault(request.name, [[], []])
        target[0].append(chunk.clone())
        target[1].append(wanted[request.offset : request.offset + chunk.numel()])

    executor.export_tensors(collect, chunk_bytes=8192)
    # Name every tensor the reference lacks at once; the Executor exports the
    # whole Program, and the reference must account for all of it.
    assert not missing, f"the reference has no values for {sorted(missing)}"
    moments = 0
    for name, (chunks, references) in values.items():
        actual, wanted = torch.cat(chunks), torch.cat(references)
        if name.endswith((".exp_avg", ".exp_avg_sq")):
            assert actual.dtype == torch.float32
            relative = (actual - wanted).norm() / wanted.norm().clamp_min(1e-20)
            assert relative <= 0.03, (name, float(relative))
            moments += 1
        else:
            torch.testing.assert_close(
                actual,
                wanted.to(actual.dtype),
                rtol=0.01 if name.startswith("master.") else 0.03,
                atol=0.0003 if name.startswith("master.") else 0.002,
                msg=lambda error, name=name: f"{name}: {error}",
            )
    assert moments and any(name.startswith("master.") for name in values)


def execute(root, request, group):
    plan = Plan.load(root / "plan.ssp")
    expected = torch.load(
        root / f"reference-{group.logical_rank}.pt", weights_only=True
    )
    with Executor(plan, device=0, rank_map=group.rank_map) as executor:
        setup = experiment(request, logical_rank=group.logical_rank)
        executor.initialize(setup["fill_tensor"], chunk_bytes=8192)
        inputs = setup["candidates"]["4"]
        records = []
        for index, reference in enumerate(expected):
            pending = executor.run(
                **inputs, hyperparams=setup["hyperparams"], runtime_trace=index == 0
            )
            result = pending.result()
            # Release the device outputs before any assertion: the Runtime
            # cannot close while caller-owned outputs still reference its pools.
            losses = [loss.cpu() for loss in result.losses]
            diagnostics = pending.diagnostics
            del result, pending
            for loss, wanted in zip(losses, reference["losses"], strict=True):
                torch.testing.assert_close(loss, wanted, rtol=0.02, atol=0.002)
            if index == 0:
                check_trace(plan, diagnostics)
            compare(executor, reference["values"])
            if index == 0:
                checkpoint = root / f"checkpoint-{group.logical_rank}.ssp"
                save_checkpoint(executor, checkpoint, chunk_bytes=8192)
                executor.initialize(lambda request, chunk: chunk.zero_(), restore=True)
                load_checkpoint(executor, checkpoint, chunk_bytes=4096)
                compare(executor, reference["values"])
            records.append({"step": index + 1, "loss": sum(float(v) for v in losses)})
        (root / f"result-{group.logical_rank}.json").write_text(
            json.dumps({"passed": True, "steps": records}, indent=2)
        )


if __name__ == "__main__":
    root = Path(sys.argv[1])
    request = read(root / "request.json")
    with workers(request, {"world_size": 2}) as group:
        execute(root, request, group)
