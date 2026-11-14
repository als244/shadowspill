"""Saved recipe execution must not prepare, compile or tune task graphs."""

import sys

from triton.runtime.autotuner import Autotuner

from engine_tests.execution._cold import forbid_preparation, forbidden

forbid_preparation()
# Keep the provider's source-verified benchmark wrapper intact. Refuse the
# actual timing callback, including an explicitly configured benchmarker.
Autotuner.do_bench = property(lambda self: forbidden)

from shadowspill.training.preparation.execute import run  # noqa: E402

run(sys.argv[1])
