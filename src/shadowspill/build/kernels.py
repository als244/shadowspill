"""Finalize Inductor kernels on valid task inputs and save their selected binaries.

Standalone compilation normally launches each kernel with unrelated random
buffers. That is unsafe for indexing through derived controls such as cumulative
sequence lengths. Compile without those launches, then tune the complete task
before build returns. Profiling and execution load the frozen result.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from threading import RLock
from typing import Any
from unittest.mock import patch

from torch._inductor.runtime.triton_heuristics import CachingAutotuner
from torch._inductor.triton_bundler import TritonBundler

_active: ContextVar[dict[str, Any] | None] = ContextVar(
    "inductor_kernels", default=None
)
_lock = RLock()


@contextmanager
def capture_kernels() -> Iterator[dict[str, Any]]:
    """Observe this task's actual launches without changing autotuner behavior."""
    with _lock:
        original: Callable[..., Any] = CachingAutotuner.run

        @wraps(original)
        def run(tuner: Any, *args: Any, **kwargs: Any) -> Any:
            result = original(tuner, *args, **kwargs)
            if (kernels := _active.get()) is not None:
                kernels[tuner.inductor_meta["kernel_name"]] = tuner
            return result

        kernels: dict[str, Any] = {}
        token = _active.set(kernels)
        try:
            with patch.object(CachingAutotuner, "run", run):
                yield kernels
        finally:
            _active.reset(token)


def freeze_kernels(function: Any, kernels: dict[str, Any]) -> dict[str, str]:
    """Replace provisional bundles with only the kernels selected by this task."""
    entry = function.inner_fn.compiled_fn.serialize()
    compiled = entry.compiled_fw.result
    original = compiled._triton_bundle
    if original is None:
        if kernels:
            raise ValueError("compiled task has kernels but no portable binary bundle")
        return {}
    selected: dict[str, str] = {}
    TritonBundler.begin_compile()
    try:
        for saved in original.static_autotuners:
            tuner = kernels.get(saved.kernel_name)
            if tuner is None:
                raise ValueError(f"build did not execute kernel {saved.kernel_name}")
            (launcher,) = tuner.launchers
            tuner._prune_compile_results_to_launcher(launcher)
            selected[saved.kernel_name] = launcher.cache_hash
            TritonBundler.put(launcher.cache_hash, tuner.triton_meta["device"])
            TritonBundler.put_winner(launcher.cache_hash)
            TritonBundler.put_static_autotuner(saved.cache_key, tuner)
        bundle, _ = TritonBundler.collect()
    finally:
        TritonBundler.end_compile()
    compiled.set_triton_bundle(bundle)
    return selected
