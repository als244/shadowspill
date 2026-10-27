"""Saved recipe execution must not prepare, compile or tune task graphs."""

import sys

from triton.runtime.autotuner import Autotuner

from tests.shadowspill.execution._cold import forbid_preparation, forbidden

forbid_preparation()
Autotuner._bench = forbidden

from workloads.recipes.text.glm53_flash_lora.execute import run  # noqa: E402

run(sys.argv[1])
