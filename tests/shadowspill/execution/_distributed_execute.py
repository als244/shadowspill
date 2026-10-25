"""Fresh NCCL worker: saved Plans only, bounded shards, eager reference checks."""

import importlib.abc
import os
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist


class NoPreparation(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {
            "shadowspill.build",
            "shadowspill.profiling.measure",
            "shadowspill.communication.measure",
            "shadowspill.store",
        }:
            raise AssertionError(f"execution imported preparation phase: {fullname}")


sys.meta_path.insert(0, NoPreparation())
from shadowspill import Executor, Plan, load_checkpoint, save_checkpoint  # noqa: E402

directory = Path(sys.argv[1])
fixture = torch.load(directory / "oracle.pt", weights_only=True)
plan = Plan.load(directory / "plan.ssp")
evaluation = directory / "evaluation-plan.ssp"
plans = {"evaluation": Plan.load(evaluation)} if evaluation.exists() else None
dist.init_process_group("gloo", timeout=timedelta(seconds=90))
logical = 1 - dist.get_rank()
assert not torch.cuda.is_initialized(), "worker initialized CUDA before Executor"


def fill(request, chunk):
    assert chunk.device.type == "cpu" and chunk.nbytes <= 40
    source = fixture["initial"][request.name].reshape(-1)
    chunk.copy_(source[request.offset : request.offset + chunk.numel()])


executor = None
try:
    executor = Executor(
        plan, plans=plans, device=int(os.environ["LOCAL_RANK"]), rank_map={0: 1, 1: 0}
    )
    executor.initialize(fill, chunk_bytes=40)
    for index, expected in enumerate(fixture["expected"]):
        calls = fixture["examples"][logical]
        pending = executor.run(
            args=[
                args
                for args, _ in calls[: plan.program.bindings.get("num_examples", 1)]
            ],
            kwargs=(
                [kwargs for _, kwargs in calls]
                if plan.program.bindings["kind"] != "forward"
                else None
            ),
            accumulate_gradients=expected["accumulate"],
        )
        output = pending.result()
        values = (
            [output] if plan.program.bindings["kind"] == "forward" else output.losses
        )
        for actual, reference in zip(values, expected["outputs"][logical], strict=True):
            torch.testing.assert_close(actual.cpu(), reference, atol=0.008, rtol=0.025)

        seen = set()

        def compare(request, chunk, seen=seen, expected=expected):
            assert chunk.nbytes <= 28
            seen.add(request.name)
            wanted = expected["state"][request.name].reshape(-1)
            torch.testing.assert_close(
                chunk,
                wanted[request.offset : request.offset + chunk.numel()],
                atol=0.004,
                rtol=0.03,
            )

        executor.export_tensors(compare, chunk_bytes=28)
        assert seen == expected["state"].keys(), (seen, expected["state"].keys())
        if index == 0:
            checkpoint = directory / f"checkpoint-rank-{logical}.ssp"
            save_checkpoint(
                executor, checkpoint, metadata={"completed": 1}, chunk_bytes=40
            )
            executor.initialize(lambda request, chunk: chunk.zero_(), restore=True)
            assert load_checkpoint(executor, checkpoint, chunk_bytes=28) == {
                "completed": 1
            }
            executor.export_tensors(compare, chunk_bytes=28)
        if plans:
            evaluated = executor.run(plan="evaluation", args=[calls[0][0]]).result()
            torch.testing.assert_close(
                evaluated.cpu(), expected["evaluation"][logical], atol=0.008, rtol=0.025
            )
            del evaluated
        print(
            f"logical rank {logical}: invocation {index} values/shards passed",
            flush=True,
        )
        del pending, output, values, actual
    executor.close()
    executor = None
finally:
    if executor is not None:
        executor.close()
    dist.destroy_process_group()
