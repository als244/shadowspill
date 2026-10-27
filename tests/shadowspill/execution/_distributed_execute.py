"""Fresh NCCL worker: saved Plans only, bounded shards, eager reference checks."""

import os
import sys
from datetime import timedelta
from pathlib import Path

# Every worker loads code compiled for process-local cuda:0. Select its physical
# GPU before importing Torch; the logical-rank permutation is independent.
local_rank = int(os.environ["LOCAL_RANK"])
visible = os.environ.get("CUDA_VISIBLE_DEVICES")
os.environ["CUDA_VISIBLE_DEVICES"] = (
    visible.split(",")[local_rank] if visible is not None else str(local_rank)
)

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from tests.shadowspill.execution._cold import forbid_preparation  # noqa: E402

forbid_preparation("shadowspill.communication.measure")
from shadowspill import Executor, Plan, load_checkpoint, save_checkpoint  # noqa: E402

directory = Path(sys.argv[1])
fixture = torch.load(directory / "oracle.pt", weights_only=True)
plan = Plan.load(directory / "plan.ssp")
evaluation = directory / "evaluation-plan.ssp"
plans = {"evaluation": Plan.load(evaluation)} if evaluation.exists() else None
dist.init_process_group("gloo", timeout=timedelta(seconds=90))
logical = 1 - dist.get_rank()
assert not torch.cuda.is_initialized(), "worker initialized CUDA before Executor"
if "initial_by_rank" in fixture:
    fixture["initial"] = fixture["initial_by_rank"][logical]
    for item in fixture["expected"]:
        item["state"] = item["states"][logical]


def fill(request, chunk):
    assert chunk.device.type == "cpu" and chunk.nbytes <= 40
    source = fixture["initial"][request.name].reshape(-1)
    chunk.copy_(source[request.offset : request.offset + chunk.numel()])


def tolerance(name=""):
    if fixture.get("precision") == "float32":
        return dict(atol=3e-6, rtol=3e-5)
    if plan.program.bindings["kind"] != "step" or name.startswith("master."):
        return dict(atol=3e-6, rtol=3e-5)
    if name.startswith("optimizer."):
        return dict(atol=2e-5, rtol=0.03)
    return dict(atol=0.004, rtol=0.025)


executor = None
try:
    executor = Executor(
        plan,
        plans=plans,
        device=0,
        rank_map={0: 1, 1: 0},
        parameter_norms=plan.program.bindings["kind"] == "step",
    )
    executor.initialize(fill, chunk_bytes=40)
    for index, expected in enumerate(fixture["expected"]):
        calls = fixture["examples"][logical]
        if fixture.get("diagnostics") and index in {0, 1}:
            from tests.shadowspill.execution._distributed_diagnostics import check

            check(executor, calls, directory / f"diagnostics-{index}")
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
            collect_parameter_norms=bool(expected["parameter_metrics"]),
        )
        output = pending.result()
        parameter_metrics = getattr(output, "parameter_metrics", {})
        values = (
            [output] if plan.program.bindings["kind"] == "forward" else output.losses
        )
        for actual, reference in zip(values, expected["outputs"][logical], strict=True):
            torch.testing.assert_close(actual.cpu(), reference, **tolerance())
        del pending, output, values, actual
        if plan.program.bindings["kind"] == "step":
            torch.testing.assert_close(
                parameter_metrics,
                expected["parameter_metrics"],
                atol=3e-5,
                rtol=3e-4,
            )

        seen = set()
        failures = []

        def compare(
            request, chunk, seen=seen, expected=expected, failures=failures, index=index
        ):
            assert chunk.nbytes <= 28
            seen.add(request.name)
            wanted = expected["state"][request.name].reshape(-1)
            try:
                wanted = wanted[request.offset : request.offset + chunk.numel()]
                if request.name in fixture.get("exact_names", ()):
                    torch.testing.assert_close(
                        chunk.view(torch.uint8),
                        wanted.view(torch.uint8),
                        rtol=0,
                        atol=0,
                    )
                    return
                torch.testing.assert_close(
                    chunk,
                    wanted,
                    **tolerance(request.name),
                )
            except AssertionError as error:
                failures.append(
                    f"logical rank {logical}, invocation {index}, "
                    f"{request.name}[{request.offset}:"
                    f"{request.offset + chunk.numel()}]: "
                    f"{error}"
                )

        executor.export_tensors(compare, chunk_bytes=28)
        nonempty = expected["state"].keys() - set(
            fixture.get("empty_by_rank", {}).get(logical, ())
        )
        assert seen == nonempty, (seen, nonempty)
        all_failures = [None] * dist.get_world_size()
        dist.all_gather_object(all_failures, failures)
        assert not any(all_failures), all_failures
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
            all_failures = [None] * dist.get_world_size()
            dist.all_gather_object(all_failures, failures)
            assert not any(all_failures), all_failures
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
    executor.close()
    executor = None
finally:
    if executor is not None:
        executor.close()
    dist.destroy_process_group()
