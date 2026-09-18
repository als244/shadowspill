"""Run one test with a compiler cache of its own, and take it away afterwards.

ShadowSpill stores compiled task manifests under ``cache_dir()``, which is
**Inductor's** cache root -- process-global, shared by every concurrent test and
persistent across runs. Two things follow, and a test should be exposed to
neither.

**Concurrent tests share it.** Nothing about a test's own `artifact_store`
changes that, so one test can be served an entry another put there.

**Runs share it too**, which is the worse half. A manifest is keyed by the FX
graph cache key and the semantic contract digest, both settled *before* Inductor
lowers -- so a compile that differs only in a decision Inductor makes *during*
lowering collides on the same key. Whatever was stored first is what later runs
get, and a test that passes on a quiet machine can fail on a busy one, or the
reverse, for reasons that are not in the tree.

So each invocation gets a fresh directory and loses it on the way out: whatever
a test sees, it put there itself, during this run. That costs the compile time a
warm cache would have saved, which is the price of a test that means the same
thing every time it runs.

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
