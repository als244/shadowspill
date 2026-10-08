"""Run one test with fresh Inductor and Triton caches, removed afterwards.

This covers direct compiler calls and PyTorch work outside ShadowSpill planning.
The public planning APIs select compiler caches within their artifact store,
so tests must also pass a test-local temporary store to every planning call.
Neither layer should read a user's persistent cache or another test's artifacts.

    python isolated_cache.py <command> [argument ...]
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile


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
    try:
        return subprocess.call(sys.argv[1:], env=environment)
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
