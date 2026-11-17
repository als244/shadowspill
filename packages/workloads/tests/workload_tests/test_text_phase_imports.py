"""Standalone phase planning imports no tensor framework or model."""

import os
import subprocess
import sys


def test_phase_planning_import_boundary():
    result = subprocess.run(
        [
            sys.executable,
            *(["-S"] if sys.flags.no_site else []),
            "-c",
            """
import importlib
import sys
importlib.import_module("shadowspill.training.preparation.plan")
assert "torch" not in sys.modules
assert "shadowspill_workloads.models" not in sys.modules
""",
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
