"""The public offline path must work even when GPU/framework imports fail."""

import os
import subprocess
import sys
import sysconfig
from dataclasses import replace
from pathlib import Path

from shadowspill import MemoryBudgets, plan_program

from ._example import communication, options, profiled


def test_fresh_cpu_process_load_plan_replan_and_report(tmp_path):
    source = profiled()
    source = replace(
        source, program=replace(source.program, executables={"update_impl": b"opaque"})
    )
    original = plan_program(
        source,
        memory_budgets=MemoryBudgets(100, 64),
        communication_info=communication(),
        search_options=options(),
    )
    original.save(tmp_path / "input.ssp")
    root = Path(__file__).resolve().parents[3]
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        PYTHONPATH=os.pathsep.join(
            (
                str(root / "src"),
                str(root),
                sysconfig.get_path("purelib"),
            )
        ),
    )
    code = """
import importlib.abc, sys
from pathlib import Path
class NoExecutionImports(importlib.abc.MetaPathFinder):
    def find_spec(self, name, *args):
        blocked = ('torch', 'mlops', 'shadowspill.runtime', 'shadowspill.pytorch',
                   'shadowspill.compilation', 'shadowspill.store')
        if any(name == item or name.startswith(item + '.') for item in blocked):
            raise AssertionError('Offline pipeline imported ' + name)
sys.meta_path.insert(0, NoExecutionImports())
from shadowspill import (Plan, Profiles, TaskProfile, apply_profiles,
                         plan_program, write_report)
from shadowspill.planner import GenericPlanningOptions, SearchOptions
plan = Plan.load('input.ssp')
assert plan.program.executables['update_impl'] == b'opaque'
updated = plan.profiled_program.profiles.merge(Profiles(
    tasks={'update_impl': TaskProfile(duration_seconds=2e-6)},
))
result = plan_program(apply_profiles(plan.program, updated),
    memory_budgets=plan.memory_budgets, communication_info=plan.communication_info,
    search_options=SearchOptions(generic=GenericPlanningOptions(
        minimum_object_bytes_evict_eligible=0), workers=1))
assert result.timeline.makespan_ns == 4000
result.save('output.ssp')
restored = Plan.load('output.ssp')
assert restored.program.executables == plan.program.executables
write_report(restored, 'report')
assert not Path('.shadowspill').exists()
print('CPU-only archive, profile override, planning and report passed')
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "passed" in result.stdout
