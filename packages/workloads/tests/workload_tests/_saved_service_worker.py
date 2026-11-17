"""Run a saved service worker with capture, compilation and tuning forbidden."""

import sys
from pathlib import Path

from triton.runtime.autotuner import Autotuner

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "engine/tests"))
from engine_tests.execution._cold import forbid_preparation, forbidden

forbid_preparation()
# Keep the provider's source-verified wrapper intact and refuse its timer.
Autotuner.do_bench = property(lambda self: forbidden)

from shadowspill.training.worker.bootstrap import main  # noqa: E402

main()
