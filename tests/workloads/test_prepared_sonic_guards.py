"""Provider cold guards must remain safe before allocator installation."""

import importlib.util
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("capability", [(9, 0), (12, 0)])
def test_target_precedes_provider_imports_and_guards_remain_active(capability):
    if any(
        importlib.util.find_spec(name) is None
        for name in ("mlops", "quack", "sonicmoe", "cutlass")
    ):
        pytest.skip("requires optional prepared Sonic provider dependencies")
    # A fresh process avoids already-imported providers concealing ordering
    # mistakes. Pretend CUDA is visible, but forbid every attempt to initialize
    # it, so this check also exercises the failure path on CPU-only hosts.
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    environment.pop("QUACK_ARCH", None)
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys, torch
from tests.workloads.prepared_sonic.guards import forbid_provider_tuning

assert not torch.cuda.is_initialized()
def forbidden_initialization(*args, **kwargs):
    raise AssertionError('CUDA initialized before allocator installation')
torch.cuda._lazy_init = forbidden_initialization
torch.cuda.is_available = lambda: True
forbid_provider_tuning(tuple(map(int, sys.argv[1:])))

from mlops.quack_runtime._benchmark_scope import install
from quack.autotuner import Autotuner as QuackAutotuner
from triton.runtime.autotuner import Autotuner as TritonAutotuner
import triton
install()
for function in (QuackAutotuner._bench, QuackAutotuner.check_disk_cache,
                 QuackAutotuner.prune_configs, TritonAutotuner._bench,
                 triton.compile):
    try:
        function(None)
    except AssertionError as error:
        assert 'provider compilation or autotuning' in str(error)
    else:
        raise AssertionError('Provider cold guard lost')
assert not torch.cuda.is_initialized()
""",
            *map(str, capability),
        ],
        env=environment,
        check=True,
        timeout=60,
        capture_output=True,
        text=True,
    )
