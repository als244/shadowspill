"""Tiny saved-workload hooks; fail if execution tries to compile or calibrate."""

from contextlib import contextmanager
from pathlib import Path

import torch

from engine_tests.execution._cold import forbid_preparation

forbid_preparation()


@contextmanager
def initializer(options, metadata, *, rank):
    initial = torch.load(Path(options["data"]) / "oracle.pt", weights_only=True)[
        "initial"
    ]

    def fill(request, destination):
        values = initial[request.name].reshape(-1)
        destination.copy_(values[request.offset : request.offset + destination.numel()])

    yield fill


def inputs(options, *, rank):
    examples = torch.load(Path(options["data"]) / "oracle.pt", weights_only=True)[
        "examples"
    ]
    return iter(
        {"args": [args for args, _ in row], "kwargs": [kwargs for _, kwargs in row]}
        for row in examples
    )


def distributed_coordination(directory):
    """Exercise real Gloo/rank mapping with a CPU-only stand-in for the GPU."""
    import os
    from contextlib import nullcontext
    from types import SimpleNamespace

    import shadowspill
    from shadowspill.execution import TrainingResult
    from tools.measure_plan import run
    from tools.measure_plan import updates as runner_updates

    rank = int(os.environ["RANK"])
    root = Path(directory)
    plan = SimpleNamespace(
        program=SimpleNamespace(
            bindings={"kind": "step"},
            parallelism=SimpleNamespace(world_size=2),
            digest="program",
        ),
        schedule=SimpleNamespace(digest="schedule"),
        timeline=SimpleNamespace(makespan_ns=1_000_000_000),
        memory_budgets=SimpleNamespace(execution_bytes=4096, spill_bytes=2048),
    )
    shadowspill.Plan.load = lambda path: plan

    class Executor:
        def __init__(self, plan, *, rank_map, parameter_norms):
            assert rank_map == {0: 1, 1: 0}
            assert parameter_norms is False
            self.plan, self.spill_capacity_bytes = plan, 128 + rank

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            assert not (root / "result.json").exists()
            (root / f"closed-{rank}").touch()

        def initialize(self, fill):
            assert fill == 1 - rank

        def run(self, **inputs):
            assert inputs == {
                "args": [(1 - rank,)],
                "runtime_trace": False,
                "hyperparams": {},
                "collect_parameter_norms": False,
            }
            return SimpleNamespace(
                result=lambda: TrainingResult((torch.tensor(1.5 + rank),), ())
            )

        def synchronize(self):
            pass

    shadowspill.Executor = Executor
    times = iter((0, 1 + rank, 0, 1 + rank))
    runner_updates.time = SimpleNamespace(perf_counter=lambda: next(times))

    result = run(
        "unused.ssp",
        output_dir=root,
        steps=2,
        monitor=False,
        trace_final_step=False,
        rank_map={0: 1, 1: 0},
        initializer=lambda context: nullcontext(context.rank),
        inputs=lambda context: [{"args": [(context.rank,)]}] * 2,
    )
    if rank == 0:
        assert result["median_seconds"] == 2
        assert [row["worker_rank"] for row in result["per_rank"]] == [1, 0]
        assert all((root / f"closed-{i}").exists() for i in range(2))
    else:
        assert result is None


if __name__ == "__main__":
    import sys

    if sys.argv[1] == "single":
        import json

        from tools.measure_plan import NsysCapture, run

        source, output = map(Path, sys.argv[2:4])
        options = json.loads(sys.argv[4])
        if options["nsys"] is not None:
            options["nsys"] = NsysCapture(**options["nsys"])
        run(
            source / "plans/training.ssp",
            output_dir=output,
            initializer=lambda context: initializer(
                {"data": source}, {}, rank=context.rank
            ),
            inputs=lambda context: inputs({"data": source}, rank=context.rank),
            **options,
        )
    else:
        distributed_coordination(sys.argv[1])
