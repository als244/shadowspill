"""Cold recipe execution must not reconstruct a model or prepare task graphs."""

import sys

from triton.runtime.autotuner import Autotuner

from tests.shadowspill.execution._cold import forbid_preparation, forbidden

forbid_preparation("workloads.models.qwen35moe.implementations")
Autotuner._bench = forbidden

from workloads.recipes.text.qwen35_lora.execute import run  # noqa: E402

run(sys.argv[1])
