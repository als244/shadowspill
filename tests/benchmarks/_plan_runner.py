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


def trainer_options(options, metadata):
    return {}


def distributed_coordination(directory):
    """Exercise real Gloo/rank mapping with a CPU-only stand-in for the GPU."""
    import os
    from contextlib import nullcontext
    from types import SimpleNamespace

    import shadowspill
    from benchmarks.plan_runner import run

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

    class Trainer:
        def __init__(self, plan, *, rank_map):
            assert rank_map == {0: 1, 1: 0}
            self.executor = SimpleNamespace(spill_capacity_bytes=128 + rank)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            assert not (root / "result.json").exists()
            (root / f"closed-{rank}").touch()

        def initialize(self, fill):
            assert fill == 1 - rank

        def fit(self, source, *, callbacks, steps, **options):
            assert list(source) == [{"args": [(1 - rank,)], "runtime_trace": False}]
            for step in range(1, steps + 1):
                callbacks[0](
                    self,
                    SimpleNamespace(
                        step=step,
                        seconds=1 + rank,
                        loss=1.5 + rank,
                        summary=SimpleNamespace(scalars={}),
                    ),
                )

    result = run(
        "unused.ssp",
        output_dir=root,
        steps=2,
        monitor=False,
        trace_final_step=False,
        rank_map={0: 1, 1: 0},
        trainer_type=Trainer,
        initializer=lambda context: nullcontext(context.rank),
        inputs=lambda context: [{"args": [(context.rank,)]}],
    )
    if rank == 0:
        assert result["median_seconds"] == 2
        assert [row["worker_rank"] for row in result["per_rank"]] == [1, 0]
        assert all((root / f"closed-{i}").exists() for i in range(2))
    else:
        assert result is None


if __name__ == "__main__":
    import sys

    distributed_coordination(sys.argv[1])
