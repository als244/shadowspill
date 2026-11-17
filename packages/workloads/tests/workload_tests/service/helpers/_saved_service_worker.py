"""Run a saved service worker with capture, compilation and tuning forbidden."""

import sys
from pathlib import Path

from triton.runtime.autotuner import Autotuner

repository = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "packages/engine/tests").is_dir()
)
sys.path.insert(0, str(repository / "packages/engine/tests"))
from engine_tests.execution._cold import forbid_preparation, forbidden  # noqa: E402

forbid_preparation()
# Keep the provider's source-verified wrapper intact and refuse its timer.
Autotuner.do_bench = property(lambda self: forbidden)

from shadowspill.training.worker.bootstrap import main  # noqa: E402

main()
