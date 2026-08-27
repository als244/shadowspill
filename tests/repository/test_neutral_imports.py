"""The neutral tree must not need a framework to import.

Producing a Program -- capture, lowering, compilation, profiling -- is the
frontend's work. Everything from a Program onwards is neutral, so planning a
saved one must not drag a framework in.

This is checked in a fresh interpreter per case: once any test in this
process has imported torch, `sys.modules` can no longer tell us who asked
for it.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

#: Each must import without torch appearing in `sys.modules`.
NEUTRAL_MODULES = (
    "shadowspill.errors",
    "shadowspill.ir",
    "shadowspill.planner",
    "shadowspill.planner.admission",
    "shadowspill.planner.plan_store",
    "shadowspill.planner.artifact_store",
    "shadowspill.planner.program",
    "shadowspill.planner.selection",
    "shadowspill.runtime",
    "shadowspill.simulator",
)

_PROBE = """
import importlib, sys
importlib.import_module({name!r})
loaded = sorted(m for m in sys.modules if m == "torch" or m.startswith("torch."))
if loaded:
    print(",".join(loaded[:8]))
    raise SystemExit(1)
"""


def _imports_without_torch(module: str) -> tuple[bool, str]:
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE.format(name=module)],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    detail = completed.stdout.strip() or completed.stderr.strip()
    return completed.returncode == 0, detail


@pytest.mark.parametrize("module", NEUTRAL_MODULES)
def test_neutral_module_imports_without_torch(module: str) -> None:
    clean, detail = _imports_without_torch(module)
    assert clean, f"{module} pulled torch in: {detail}"


def test_planning_a_saved_program_needs_no_frontend() -> None:
    """The entry point a budget sweep calls, and the worker that calls it.

    The planning-eval worker plans thousands of saved Programs and never
    executes a model. It importing torch would mean paying for a framework
    to read costs that were measured months ago.
    """

    for module in ("shadowspill.planner", "benchmarking.planning_eval.worker"):
        clean, detail = _imports_without_torch(module)
        assert clean, f"{module} pulled torch in: {detail}"
