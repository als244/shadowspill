"""Fresh saved-Plan EP worker; model construction and preparation are forbidden.

It trains from the saved Plan alone and keeps the whole state it holds after
initialization and after every step, for the reference to take each step from.
"""

import sys
from pathlib import Path

import torch

from tests.shadowspill.execution._cold import forbid_preparation

forbid_preparation()
from benchmarking.quickstart.phases.common import experiment, read  # noqa: E402
from benchmarking.quickstart.phases.distributed import workers  # noqa: E402
from shadowspill import Executor, Plan, load_checkpoint, save_checkpoint  # noqa: E402
from tests.shadowspill.execution._trace_checks import check_trace  # noqa: E402


def exported(executor):
    """The state this rank holds: each tensor's region as its offset in the
    complete tensor and its values."""
    regions = {}

    def collect(request, chunk):
        assert chunk.device.type == "cpu" and chunk.nbytes <= 8192
        offset, parts = regions.setdefault(request.name, (request.offset, []))
        assert request.offset == offset + sum(part.numel() for part in parts)
        parts.append(chunk.clone())

    executor.export_tensors(collect, chunk_bytes=8192)
    return {
        name: (offset, torch.cat(parts)) for name, (offset, parts) in regions.items()
    }


def execute(root, request, group):
    plan = Plan.load(root / "plan.ssp")
    with Executor(plan, device=0, rank_map=group.rank_map) as executor:
        setup = experiment(request, logical_rank=group.logical_rank)
        executor.initialize(setup["fill_tensor"], chunk_bytes=8192)
        inputs = setup["candidates"]["4"]
        states = [{"losses": [], "tensors": exported(executor)}]
        for index in range(request["steps"]):
            pending = executor.run(
                **inputs, hyperparams=setup["hyperparams"], runtime_trace=index == 0
            )
            result = pending.result()
            # Release the device outputs before any assertion: the Runtime
            # cannot close while caller-owned outputs still reference its pools.
            losses = [loss.cpu() for loss in result.losses]
            diagnostics = pending.diagnostics
            del result, pending
            if index == 0:
                check_trace(plan, diagnostics)
            states.append({"losses": losses, "tensors": exported(executor)})
            if index == 0:
                checkpoint = root / f"checkpoint-{group.logical_rank}.ssp"
                save_checkpoint(executor, checkpoint, chunk_bytes=8192)
                executor.initialize(lambda request, chunk: chunk.zero_(), restore=True)
                load_checkpoint(executor, checkpoint, chunk_bytes=4096)
                # A checkpoint holds the state exactly: every tensor comes
                # back bit for bit, so training resumes where it stopped.
                held, restored = states[-1]["tensors"], exported(executor)
                assert restored.keys() == held.keys()
                changed = sorted(
                    name
                    for name, (offset, values) in held.items()
                    if restored[name][0] != offset
                    or not torch.equal(restored[name][1], values)
                )
                assert not changed, f"a checkpoint round trip changed {changed}"
    torch.save(states, root / f"state-{group.logical_rank}.pt")


if __name__ == "__main__":
    root = Path(sys.argv[1])
    request = read(root / "request.json")
    with workers(request, {"world_size": 2}) as group:
        execute(root, request, group)
