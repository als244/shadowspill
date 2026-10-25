"""Cold recipe execution must not reconstruct a model or prepare task graphs."""

import importlib.abc
import sys

import torch._inductor.compile_fx as compiler
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
                "workloads.models.qwen35moe.implementations",
            )
        ):
            raise AssertionError("execution imported " + fullname)


def forbidden(*args, **kwargs):
    raise AssertionError("execution tried graph compilation or autotuning")


compiler.compile_fx = compiler.compile_fx_inner = forbidden
Autotuner._bench = forbidden
sys.meta_path.insert(0, BlockPreparation())

from workloads.recipes.text.qwen35_lora.execute import run  # noqa: E402

run(sys.argv[1])
