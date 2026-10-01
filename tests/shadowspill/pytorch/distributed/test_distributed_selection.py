from __future__ import annotations

import tempfile
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from shadowspill.ir import (
    MemoryLocation,
    ResidencySpec,
    ResourceKind,
    ResourceSpec,
    TaskProfile,
    TaskSpec,
)
from shadowspill.planner import GenericPlanningOptions, SearchOptions
from shadowspill.planner.search.algorithms.pressurefit import PressureFit
from shadowspill.pytorch.distributed import Distributed
from shadowspill.pytorch.distributed._selection import choose, restore_result
from shadowspill.simulator import simulate
from tests.shadowspill.planner._examples import config, recomputation_program


def worker(rank, root):
    dist.init_process_group(
        "gloo",
        init_method="file://" + str(Path(root, "store")),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=40),
    )
    bound = Distributed(dist.group.WORLD, timeout=30)._bind(
        torch.nn.Linear(1, 1), dist.group.WORLD, namespace="selection"
    )
    try:
        with bound.activate():
            program = recomputation_program(recompute_workspace_bytes=140)
            program = replace(
                program,
                profiles=(
                    *program.profiles,
                    TaskProfile("begin_profile", 0, 0, "begin_abi"),
                ),
                tasks=(
                    TaskSpec(
                        "begin",
                        ResourceSpec("cuda_0", ResourceKind.CONTROL),
                        "begin_profile",
                        requires_entrypoint=False,
                    ),
                    *(
                        replace(task, dependencies=("begin", *task.dependencies))
                        for task in program.tasks
                    ),
                ),
            )
            program = replace(
                program,
                profiles=tuple(
                    replace(p, runtime_ns=6000 if rank == 0 else 60)
                    if p.profile_id == "recompute_profile"
                    else replace(p, runtime_ns=60 if rank == 0 else 6000)
                    if p.profile_id == "forward_profile"
                    else p
                    for p in program.profiles
                ),
            )
            for unequal in (True, False):
                simulation = config(230 if rank and unequal else 300)
                initial = (ResidencySpec("input_storage", MemoryLocation.SPILL),)
                options = SearchOptions(
                    generic=GenericPlanningOptions(
                        minimum_object_bytes_evict_eligible=0, deterministic=True
                    )
                )

                def attempt(
                    fixed,
                    carried,
                    initial=initial,
                    simulation=simulation,
                    options=options,
                ):
                    result = PressureFit()(
                        fixed,
                        initial_residency=initial,
                        config=simulation,
                        workers=1,
                        generic=options.generic,
                        incumbent=carried,
                    )
                    return SimpleNamespace(result=result, simulation=result.simulation)

                local, key, successful, _evidence = choose(
                    program, attempt, search_options=options
                )
                result = restore_result(
                    local, program, key, successful, keep_resolutions=True
                )
                assert len(successful) == (1 if unequal else 2)
                assert result.selections and result.diagnostics.resolved_programs
                assert all(
                    item.choices for item in result.diagnostics.resolved_programs
                )
                assert len(result.resolutions) == len(successful)
                if unequal:
                    assert result.selections[0].option_id == "save"
                replay = simulate(
                    program,
                    result.schedule,
                    selections=result.selections,
                    config=simulation,
                )
                assert replay.makespan_ns == result.simulation.makespan_ns
                bound.control.agree(
                    "chosen", [(v.group_id, v.option_id) for v in result.selections]
                )
    finally:
        bound.close()
        dist.destroy_process_group()


def test_common_choices_keep_local_memory_and_diagnostics():
    with tempfile.TemporaryDirectory() as root:
        mp.spawn(worker, args=(root,), nprocs=2, join=True)
