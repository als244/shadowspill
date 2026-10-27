"""Qualify fixed provider kernels, saved execution, and resumed text training."""

import sys
from pathlib import Path

import triton
from triton.runtime.autotuner import Autotuner

from tests.shadowspill.execution._cold import forbid_preparation, forbidden

forbid_preparation("workloads.models.llama3.implementations")
Autotuner._bench = forbidden
compiled = []


def listener(**event):
    if not event["cache_hit"]:
        compiled.append(event["src"].name)


triton.knobs.compilation.listener = listener

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
