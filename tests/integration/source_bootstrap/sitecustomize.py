"""Select this canary's checkout ahead of a different editable installation."""

import sys
from pathlib import Path

root = Path(__file__).resolve().parents[3]
sys.meta_path[:] = [
    finder
    for finder in sys.meta_path
    if type(finder).__module__ != "_shadowspill_editable"
]
sys.path[:0] = [str(root / "src"), str(root)]
