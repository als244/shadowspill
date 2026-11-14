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

_active: ContextVar[dict[tuple[str, str], Any] | None] = ContextVar(
    "inductor_kernels", default=None
)
_lock = RLock()


@contextmanager
def capture_kernels() -> Iterator[dict[tuple[str, str], Any]]:
    """Observe this task's actual launches without changing autotuner behavior."""
    with _lock:
        original: Callable[..., Any] = CachingAutotuner.run

        @wraps(original)
        def run(tuner: Any, *args: Any, **kwargs: Any) -> Any:
            result = original(tuner, *args, **kwargs)
            if (kernels := _active.get()) is not None:
                kernels[_kernel_key(tuner)] = tuner
            return result

        kernels: dict[tuple[str, str], Any] = {}
        token = _active.set(kernels)
        try:
            with patch.object(CachingAutotuner, "run", run):
                yield kernels
        finally:
            _active.reset(token)


def _kernel_key(tuner: Any) -> tuple[str, str]:
    # Child graphs can reuse generated names while specializing different sizes.
    # The source hash distinguishes their binaries during parent finalization.
    return tuner.inductor_meta["kernel_name"], tuner.kernel_hash


def _composition_children(function: Any) -> dict[str, Any] | None:
    from shadowspill.pytorch.chunking.executable import ChunkedExecutable
    from shadowspill.pytorch.chunking.region_executable import RegionExecutable

    if isinstance(function, (ChunkedExecutable, RegionExecutable)):
        children = {str(key): value for key, value in function.children.items()}
        if isinstance(function, ChunkedExecutable):
            children.update(
                {
                    f"accumulate:{key}": value
                    for key, value in function.accumulating_children.items()
                }
            )
        return children
    return None


def freeze_kernels(
    function: Any, kernels: dict[tuple[str, str], Any]
) -> dict[str, str]:
    """Replace provisional bundles with only the kernels selected by this task."""
    if (children := _composition_children(function)) is not None:
        composed = {}
        for size, child in children.items():
            names = _declared_kernels(child)
            frozen = freeze_kernels(
                child, {name: tuner for name, tuner in kernels.items() if name in names}
            )
            composed.update({f"{size}:{name}": value for name, value in frozen.items()})
        return composed
    entry = function.inner_fn.compiled_fn.serialize()
    compiled = entry.compiled_fw.result
    original = compiled._triton_bundle
    if original is None:
        if kernels:
            raise ValueError("compiled task has kernels but no portable binary bundle")
        return {}
    declared = {_kernel_key(item.kernel) for item in original.static_autotuners}
    if missing := kernels.keys() - declared:
        raise ValueError(
            "Inductor did not package statically launchable kernels: "
            + ", ".join(f"{name}:{key}" for name, key in sorted(missing))
        )
    selected: dict[str, str] = {}
    TritonBundler.begin_compile()
    try:
        for saved in original.static_autotuners:
            tuner = kernels.get(_kernel_key(saved.kernel))
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
    for saved in bundle.static_autotuners:
        # Loading even a single selected binary calls precompile again. Inductor
        # can then add reduction tiles for occupancy and benchmark them on the
        # first invocation. Freeze only the saved copy, after ordinary build
        # tuning has chosen its winner; eager tuners keep their normal policy.
        tuner = saved.kernel
        tuner.inductor_meta = {
            **tuner.inductor_meta,
            "dynamic_scale_rblock": False,
            "coordinate_descent_tuning": False,
            "force_disable_caches": True,
        }
        tuner.inductor_meta.pop("combo_tuning_groups", None)
        # This eligibility predicate is a cached_property and may already be
        # True in the serialized tuner. Changing its input metadata is not enough.
        tuner._could_rblock_scale = False
    compiled.set_triton_bundle(bundle)
    return selected


def _declared_kernels(function: Any) -> set[tuple[str, str]]:
    if (children := _composition_children(function)) is not None:
        return set().union(*(_declared_kernels(child) for child in children.values()))
    saved = function.inner_fn.compiled_fn.serialize().compiled_fw.result
    return (
        {_kernel_key(item.kernel) for item in saved._triton_bundle.static_autotuners}
        if saved._triton_bundle is not None
        else set()
    )
