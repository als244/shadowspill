"""Save measured optimizer Plans and run them with no model or compiler access."""

import os
import subprocess
import sys
from pathlib import Path

from shadowspill import (
    CommunicationInfo,
    MemoryBudgets,
    apply_profiles,
    measure_program,
    plan_program,
)


def save_plan(program, directory):
    profiles = measure_program(program, warmup_seconds=0.001, measure_seconds=0.002)
    plan = plan_program(
        apply_profiles(program, profiles),
        memory_budgets=MemoryBudgets(2 << 30, 64 << 20),
        communication_info=CommunicationInfo(1e10, 1e10),
    )
    plan.save(directory / "plan.ssp")
    return plan


def run_worker(directory):
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(Path(__file__).parents[1] / "_execute.py"),
            str(directory),
        ],
        env={
            **os.environ,
            "TRITON_CACHE_DIR": str(directory / "empty-triton"),
            "TORCHINDUCTOR_CACHE_DIR": str(directory / "empty-inductor"),
        },
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Qualified three saved-plan invocations" in result.stdout
