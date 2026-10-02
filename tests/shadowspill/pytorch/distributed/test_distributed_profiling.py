from __future__ import annotations

import tempfile
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from shadowspill.profiling.store import ProfileStore
from shadowspill.profiling.timing import collect_timing_samples, condition_task
from shadowspill.pytorch.distributed import Distributed
from shadowspill.pytorch.distributed._profiling import (
    continue_timing,
    invocation,
    phase,
)
from shadowspill.pytorch.profiling.profiler.measurement import (
    warm_persistent_allocations,
)
from shadowspill.pytorch.profiling.runner import profile_unique_artifacts
from shadowspill.task.profiles import ProfileEnvironment, TaskMeasurement
from shadowspill.task.profiling import ProfilingOptions


def worker(rank, root):
    dist.init_process_group(
        "gloo",
        init_method="file://" + str(Path(root, "rendezvous")),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    bound = Distributed(dist.group.WORLD, timeout=20)._bind(
        torch.nn.Linear(1, 1), dist.group.WORLD, namespace="profile"
    )
    try:
        count = 0

        def collective():
            nonlocal count
            with invocation():
                value = torch.tensor(rank + 1)
                dist.all_reduce(value)
                assert int(value) == 3
                count += 1
            return 1000 if rank else 100

        options = ProfilingOptions(
            conditioning_seconds=0.0000005,
            conditioning_wall_seconds=1,
            measurement_seconds=0.0000005,
            measurement_wall_seconds=1,
            minimum_samples=3,
        )
        env = ProfileEnvironment("test", None, "cpu", (0, 0), "cpu-test", "test")
        cache = ProfileStore(Path(root, "profiles"))
        artifacts = [
            SimpleNamespace(kind="forward", compatibility_digest=key)
            for key in ("a", "b", "a")
        ]

        def measure(artifact):
            readings = iter([0, 1, 1, 1, 1] if rank else [0, 1, 2, 3, 3])
            with phase("warmup"):
                warm_persistent_allocations(
                    collective, lambda: next(readings), 1, stabilization_iterations=5
                )
            with phase("conditioning"):
                condition_task(
                    collective, options=options, continue_while=continue_timing
                )
            with phase("timing"):
                timing = collect_timing_samples(
                    collective, options=options, continue_while=continue_timing
                )
            assert len(timing.samples) == 5
            return TaskMeasurement(
                timing.samples[0], 0, 0, (), timing.samples, "cpu-collective"
            )

        with bound.activate():
            first = profile_unique_artifacts(
                artifacts, environment=env, measure=measure, cache=cache
            )
            assert first.unique_keys == 2 and first.cache_misses == 2
            assert first.measurements[0] == first.measurements[2]
            assert count == 2 * (4 + 5 + 5)
            before = count
            second = profile_unique_artifacts(
                artifacts, environment=env, measure=measure, cache=cache
            )
            assert second.cache_hits == 2 and count == before
            # Remove exactly one rank's first entry. Both ranks must execute its
            # collectives again; the valid peer record stays unchanged.
            if rank:
                digest = first.key_digests[0]
                for path in Path(root, "profiles").rglob("measurement.json"):
                    import json

                    if json.loads(path.read_text())["key_digest"] == digest:
                        path.unlink()
            bound.control.exchange("removed-cache", True)
            third = profile_unique_artifacts(
                artifacts, environment=env, measure=measure, cache=cache
            )
            assert third.cache_hits == 1 and third.cache_misses == 1
            assert count - before == 14
            assert third.measurements == first.measurements
            # No profiling control calls are inserted into subsequent model work.
            sequence = bound.control.sequence
            value = torch.tensor(rank + 1)
            dist.all_reduce(value)
            assert int(value) == 3 and bound.control.sequence == sequence
    finally:
        bound.close()
        dist.destroy_process_group()


def test_actual_profiler_coordinates_warmups_and_partial_cache():
    with tempfile.TemporaryDirectory() as root:
        mp.spawn(worker, args=(root,), nprocs=2, join=True)


def _geometry_failure_worker(rank, root):
    from unittest.mock import patch

    from shadowspill.errors import ProfilingError
    from shadowspill.planner import StepDataOrdering
    from shadowspill.pytorch.failures import device_exhausted
    from shadowspill.pytorch.step_search.sweep import _Build

    dist.init_process_group(
        "gloo",
        init_method="file://" + str(Path(root, "geometry-rendezvous")),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    model = torch.nn.Linear(1, 1)
    bound = Distributed(dist.group.WORLD, timeout=20)._bind(
        model, dist.group.WORLD, namespace="geometry-recovery"
    )

    class LocalRuntime:
        last_failure = None

    build = _Build(
        model=model,
        objective=None,
        optimizer=None,
        hyperparams=(),
        candidates={name: [(name,)] for name in ("early", "invoke", "small", "fatal")},
        runtime=LocalRuntime(),
        execution="execution",
        spill="spill",
        optimizer_ordering="stage_interleaved",
        verbose=False,
        artifact_store=None,
        build_store=None,
        build_store_mode="contribute",
        export_bypass_key=None,
        master_dtype=None,
        grad_dtype=None,
        round_accumulation_once=False,
        memory_bound_flops_per_byte=1,
    )
    calls = []

    def wrapped_oom():
        raise ProfilingError("wrapped allocation failure") from torch.OutOfMemoryError(
            "test device exhaustion"
        )

    def fake_build(*args, **kwargs):
        name = kwargs["example_inputs"][0][0]
        calls.append(name)
        if name == "early" and rank:
            wrapped_oom()
        if name == "invoke":

            def measure():
                if rank:
                    wrapped_oom()

            bound.control.run("measurement/test/measure", measure)
        if name == "fatal":
            with invocation():
                # Both ranks must reach the body before either injected error
                # can interrupt the peer's preparation boundary.
                dist.barrier()
                if rank:
                    raise ValueError("invalid operator contract")
                wrapped_oom()
        if name == "small":
            with invocation():
                value = torch.tensor(rank + 1)
                dist.all_reduce(value)
                assert value.item() == 3
        return tuple(object() for _ in kwargs["orderings"])

    orderings = (StepDataOrdering.depth_first(1),)
    try:
        with (
            bound.activate(),
            patch(
                "shadowspill.pytorch.step_search.sweep.build_step_programs", fake_build
            ),
        ):
            parent = bound.control
            for name in ("early", "invoke"):
                programs, exhausted = build.programs(name, 1, orderings)
                assert programs == () and exhausted is not None
                assert device_exhausted(exhausted.error)
                assert "wrapped allocation failure" in str(exhausted.error)
                assert bound.control is parent
                # A real collective in the next geometry must still work even
                # when only one rank failed, including before profiling began.
                programs, exhausted = build.programs("small", 1, orderings)
                assert len(programs) == 1 and exhausted is None
                assert bound.control is parent
            try:
                build.programs("fatal", 1, orderings)
            except Exception as error:
                assert "invalid operator contract" in str(error)
                assert not device_exhausted(error)
            else:
                raise AssertionError("a non-OOM error was incorrectly skipped")
            assert calls == ["early", "small", "invoke", "small", "fatal"]
    finally:
        bound.close()
        dist.destroy_process_group()


def test_geometry_oom_recovers_collectively_without_hiding_other_errors():
    with tempfile.TemporaryDirectory() as root:
        mp.spawn(_geometry_failure_worker, args=(root,), nprocs=2, join=True)
