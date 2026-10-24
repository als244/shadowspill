"""One GPU can build/measure a symmetric multi-rank forward Program."""

import os
import subprocess
import sys

import pytest
import torch
from torch import nn

from shadowspill import (
    Parallelism,
    ProfiledProgram,
    apply_profiles,
    build_forward,
    measure_program,
)


@pytest.mark.cuda
def test_build_profile_locally_then_plan_sharded_forward_on_cpu(tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("requires representative GPU")

    def forbidden(*args, **kwargs):
        raise AssertionError("local DP tasks must not construct process groups")

    monkeypatch.setattr(torch.distributed, "init_process_group", forbidden)
    model = nn.Linear(9, 7, device="meta")
    program = build_forward(
        model,
        example_args=(torch.empty(3, 9),),
        parallelism=Parallelism.dp(ranks_per_replica=8),
    )
    assert program.parallelism.world_size == 8
    shards = program.bindings["weight_shards"]
    assert sorted(item["shard_bytes"] for item in shards.values()) == [4, 32]
    assert sorted(item["padded_bytes"] for item in shards.values()) == [32, 256]
    profiles = measure_program(program, warmup_seconds=0.005, measure_seconds=0.01)
    profiled = apply_profiles(program, profiles)
    assert not profiled.estimated_memory
    profiled.save(tmp_path / "profiled.ssp")
    assert (
        ProfiledProgram.load(tmp_path / "profiled.ssp").program.parallelism
        == program.parallelism
    )
    code = """
import importlib.abc, sys
class CPUOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, name, *args):
        if (name.split('.')[0] in ('torch', 'mlops')
            or name.startswith('shadowspill.runtime')):
            raise AssertionError('CPU planning imported ' + name)
sys.meta_path.insert(0, CPUOnly())
from shadowspill import (ProfiledProgram, CommunicationInfo, MemoryBudgets,
                        plan_program, write_report)
profiled = ProfiledProgram.load(sys.argv[1])
plan = plan_program(profiled, memory_budgets=MemoryBudgets(1 << 30, 1 << 20),
    communication_info=CommunicationInfo(25e9, 25e9,
        topology={'node0': tuple(range(8))}, scale_up_bytes_per_second=300e9,
        scale_up_latency_seconds=2e-6))
assert plan.summary['logical_ranks'] == 8
assert plan.summary['simulation_only']
assert plan.memory_layout is None
assert any(a.transfer and len(a.transfer.fetch) == 2
           for a in plan._costed_program.alias_groups)
plan.save(sys.argv[2])
write_report(plan, sys.argv[3])
print('single-GPU build/measurement and CPU-only sharded simulation passed')
"""
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            code,
            str(tmp_path / "profiled.ssp"),
            str(tmp_path / "plan.ssp"),
            str(tmp_path / "report"),
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "simulation passed" in result.stdout
