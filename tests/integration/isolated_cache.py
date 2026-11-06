"""Run one test with fresh Inductor and Triton caches, removed afterwards.

This covers direct compiler calls and PyTorch work outside ShadowSpill planning.
Build uses this compiler cache root; profiling restores its finalized code.
Tests save Programs, Profiles and Plans in temporary directories. Neither layer
should read a user's persistent cache or another test's artifacts.

    python isolated_cache.py <command> [argument ...]
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: isolated_cache.py <command> [argument ...]", file=sys.stderr)
        return 2
    root = tempfile.mkdtemp(prefix="shadowspill-test-cache-")
    environment = dict(os.environ)
    environment["TORCHINDUCTOR_CACHE_DIR"] = root
    # Triton keeps its own directory and reads this one; naming it here means a
    # test cannot inherit a kernel another test compiled.
    environment["TRITON_CACHE_DIR"] = os.path.join(root, "triton")
    # CTest may use an interpreter with a scikit-build editable install of a
    # different checkout. Its meta-path finder precedes PYTHONPATH, so naming
    # this tree in PYTHONPATH alone does not select its Python implementation.
    # Keep this bootstrap inherited by every fresh worker the canary starts.
    bootstrap = Path(__file__).with_name("source_bootstrap")
    inherited = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(bootstrap), *([inherited] if inherited else [])]
    )
    try:
        return subprocess.call(sys.argv[1:], env=environment)
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
