"""The top-level package exposes the phase API through explicit lazy imports.

Whether importing it drags a framework in is checked in
`test_neutral_imports.py`, which runs each case in a fresh interpreter. The
same check cannot be made here: once any earlier test in this process has
imported torch, a `sys.modules` diff can no longer attribute it.
"""

from __future__ import annotations

import importlib
import subprocess
import sys


def test_top_level_public_surface_is_explicit() -> None:
    package = importlib.import_module("shadowspill")
    assert set(package.__all__) == {
        "__version__",
        "build_forward",
        "build_forward_backward",
        "build_step",
        "backward_program",
        "optimizer_program",
        "instantiate_examples",
        "Program",
        "Profiles",
        "ProfilingOptions",
        "PoolMemoryProfile",
        "ProfiledProgram",
        "Plan",
        "TaskProfile",
        "TaskProfileCache",
        "OverlapProfileCache",
        "SweepResult",
        "apply_profiles",
        "measure_program",
        "measure_pool_memory",
        "MemoryBudgets",
        "CommunicationInfo",
        "measure_communication",
        "measure_overlap",
        "plan_program",
        "sweep_programs",
        "write_report",
        "Executor",
        "TrainingResult",
        "TensorChunk",
        "save_checkpoint",
        "load_checkpoint",
        "Trainer",
        "Parallelism",
        "ParameterStorage",
        "Replicated",
        "Sharded",
        "GroupRef",
        "Resource",
        "OperatorPreparation",
    }
    assert len(package.__all__) == len(set(package.__all__))
    assert package.__version__


def test_frontend_tuning_does_not_import_external_operation_providers():
    script = """
import importlib.abc, sys
class NoProvider(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'quack', 'mlops', 'sonicmoe'}:
            raise AssertionError('implicit provider import: ' + fullname)
sys.meta_path.insert(0, NoProvider())
from shadowspill.pytorch.tuning import capture_choices, use_choices
with capture_choices() as choices:
    pass
assert choices == {}
assert use_choices(lambda: 7, choices)() == 7
roots = {name.split('.')[0] for name in sys.modules}
assert not roots.intersection({'quack', 'mlops', 'sonicmoe'})
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=45
    )
    assert result.returncode == 0, result.stdout + result.stderr
