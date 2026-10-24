"""Qualify fixed provider kernels, saved execution, and resumed text training."""

import importlib.abc
import sys
from pathlib import Path

import torch._inductor.compile_fx as compiler
import triton
from triton.runtime.autotuner import Autotuner


class BlockPreparation(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(
            fullname == key or fullname.startswith(key + ".")
            for key in (
                "shadowspill.build",
                "shadowspill.profiling.measure",
                "shadowspill.planning.plan",
                "shadowspill.store",
                "shadowspill.training._legacy_trainer",
                "shadowspill.training.backends",
                "workloads.models.llama3.implementations",
            )
        ):
            raise AssertionError("execution imported " + fullname)


def forbidden(*args, **kwargs):
    raise AssertionError("execution tried graph compilation or autotuning")


compiler.compile_fx = compiler.compile_fx_inner = forbidden
Autotuner._bench = forbidden
compiled = []


def listener(**event):
    if not event["cache_hit"]:
        compiled.append(event["src"].name)


triton.knobs.compilation.listener = listener
sys.meta_path.insert(0, BlockPreparation())

from shadowspill import Trainer  # noqa: E402
from workloads.recipes.text.llama3.execute import run  # noqa: E402

original_step = Trainer.step
first_step = True


def checked_step(self, **kwargs):
    global first_step
    before = len(compiled)
    result = original_step(self, **kwargs)
    if not first_step:
        assert len(compiled) == before, compiled[before:]
    first_step = False
    return result


Trainer.step = checked_step
root = Path(sys.argv[1])
run(root)
assert set(compiled) == {
    "_forward_kernel",
    "_backward_kernel",
    "_rope_table_kernel",
    "_cross_entropy_kernel",
    "_adamw_torch_kernel",
    "_segment_kernel",
}, compiled
print("Qualified cold text execution; fixed kernels:", compiled, flush=True)
