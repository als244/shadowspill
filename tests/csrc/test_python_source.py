"""CTest workers must pair their native libraries with their own checkout."""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP = ROOT / "tests/integration/source_bootstrap/sitecustomize.py"


def test_canary_bootstrap_overrides_only_the_shadowspill_editable_finder():
    code = f"""
import runpy, sys
class WrongCheckout:
    __module__ = '_shadowspill_editable'
    def find_spec(self, fullname, *args):
        if fullname == 'shadowspill':
            raise AssertionError('selected the installed checkout')
class Unrelated:
    def find_spec(self, *args):
        return None
wrong, unrelated = WrongCheckout(), Unrelated()
sys.meta_path[:0] = [wrong, unrelated]
runpy.run_path({str(BOOTSTRAP)!r})
assert wrong not in sys.meta_path and unrelated in sys.meta_path
import shadowspill
assert shadowspill.__file__.startswith({str(ROOT / "packages/engine/src")!r})
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=30)


def test_canary_wrapper_inherits_source_selection_and_removes_caches(tmp_path):
    code = """
import json, os, subprocess, sys
import shadowspill
child = subprocess.check_output(
    [sys.executable, '-c', 'import shadowspill; print(shadowspill.__file__)'],
    text=True,
).strip()
print(json.dumps({'parent': shadowspill.__file__, 'child': child,
                  'cache': os.environ['TORCHINDUCTOR_CACHE_DIR']}))
"""
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tests/integration/isolated_cache.py"),
            sys.executable,
            "-c",
            code,
        ],
        cwd=tmp_path,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    observed = json.loads(result.stdout)
    assert Path(observed["parent"]).is_relative_to(ROOT / "packages/engine/src")
    assert observed["child"] == observed["parent"]
    assert not Path(observed["cache"]).exists()
