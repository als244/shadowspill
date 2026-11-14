"""Enforce the saved-worker boundary without importing any model definition."""

import importlib.abc
import sys


class _BlockPreparation(importlib.abc.MetaPathFinder):
    def __init__(self, extra):
        self.names = (
            "shadowspill.build",
            "shadowspill.store",
            "shadowspill.profiling.measure",
            "shadowspill.planning.plan",
            "shadowspill.pytorch.capture",
            *extra,
        )

    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == n or fullname.startswith(n + ".") for n in self.names):
            raise AssertionError(f"execution imported {fullname}")


def forbidden(*args, **kwargs):
    raise AssertionError("execution tried to compile or calibrate")


def forbid_preparation(*extra):
    import torch._inductor.compile_fx as compiler

    from shadowspill.runtime import Runtime

    compiler.compile_fx = compiler.compile_fx_inner = Runtime._calibrate = forbidden
    sys.meta_path.insert(0, _BlockPreparation(extra))
