"""The top-level package exposes the phase API through explicit lazy imports.

Whether importing it drags a framework in is checked in
`test_neutral_imports.py`, which runs each case in a fresh interpreter. The
same check cannot be made here: once any earlier test in this process has
imported torch, a `sys.modules` diff can no longer attribute it.
"""

from __future__ import annotations

import importlib


def test_top_level_public_surface_is_explicit() -> None:
    package = importlib.import_module("shadowspill")
    assert set(package.__all__) == {
        "__version__",
        "build_forward",
        "build_forward_backward",
        "build_step",
        "Program",
        "Profiles",
        "ProfiledProgram",
        "Plan",
        "TaskProfile",
        "SweepResult",
        "apply_profiles",
        "measure_program",
        "MemoryBudgets",
        "CommunicationInfo",
        "measure_communication",
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
    }
    assert len(package.__all__) == len(set(package.__all__))
    assert package.__version__
