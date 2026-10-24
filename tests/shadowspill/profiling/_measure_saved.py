"""Fresh worker: saved code must measure without build or runtime initialization."""

import importlib.abc
import sys

import torch
import torch._inductor.compile_fx as compiler
import triton

from shadowspill import Profiles, Program, measure_program


def no_compilation(*args, **kwargs):
    raise AssertionError("measurement tried to compile a finalized task")


class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(
            fullname == name or fullname.startswith(name + ".")
            for name in (
                "shadowspill.runtime",
                "shadowspill.pipeline",
                "shadowspill.store",
                "shadowspill.pytorch.compilation",
                "shadowspill.build",
            )
        ):
            raise AssertionError(
                f"profiling imported execution/build module {fullname}"
            )


compiler.compile_fx = compiler.compile_fx_inner = triton.compile = no_compilation
sys.meta_path.insert(0, BlockRuntime())
program = Program.load(sys.argv[1])
profiles = measure_program(program, warmup_seconds=0, measure_seconds=0.001)
profiles.save(sys.argv[2])
assert Profiles.load(sys.argv[2]).tasks.keys() == profiles.tasks.keys()
assert torch.cuda.memory.get_allocator_backend() == "native"
