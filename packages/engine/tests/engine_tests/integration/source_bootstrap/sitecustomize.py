"""Select this canary's checkout ahead of a different editable installation."""

import sys
from pathlib import Path

engine = Path(__file__).resolve().parents[4]
root = engine.parents[1]
sys.meta_path[:] = [
    finder
    for finder in sys.meta_path
    if type(finder).__module__ != "_shadowspill_editable"
]
sys.path[:0] = [
    str(engine / "src"),
    str(engine / "tests"),
    str(root / "packages/workloads/src"),
    str(root / "packages/workloads/tests"),
    str(root),
]
