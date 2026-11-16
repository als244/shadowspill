"""Run the installed worker with compilation/capture/planning forbidden."""

import sys
from pathlib import Path

import triton
from triton.runtime.autotuner import Autotuner

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "engine/tests"))
from engine_tests.execution._cold import forbid_preparation, forbidden

forbid_preparation("shadowspill_workloads.models.llama3.implementations")
Autotuner._bench = forbidden
compiled = set()


def listener(**event):
    if not event["cache_hit"]:
        name = event["src"].name
        assert name in {
            "_forward_kernel",
            "_backward_kernel",
            "_rope_table_kernel",
            "_segment_kernel",
            "_adamw_torch_kernel",
        }, name
        compiled.add(name)


triton.knobs.compilation.listener = listener
from shadowspill.training.worker.bootstrap import main  # noqa: E402

main()
# Fixed provider kernels can initialize on a cold worker, as in the previous
# Llama execution qualification. No graph/task compilation or autotuning occurs.
print(
    "Qualified cold Llama worker; fixed provider kernels:", sorted(compiled), flush=True
)
