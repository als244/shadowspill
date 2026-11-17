"""Fail if an exact reused case rebuilds or remeasures in a real GPU phase."""

import runpy
import sys
from pathlib import Path

from shadowspill.training.preparation.job import JOB_NAME, PreparationJob


def main():
    module, directory = sys.argv[1:]
    config = PreparationJob.load(Path(directory) / JOB_NAME).config
    if config.preparation.reuse:
        import shadowspill
        from shadowspill.training.preparation import capture, profile

        def forbidden(*args, **kwargs):
            raise AssertionError(
                "exact measurement reuse attempted build or measurement"
            )

        capture.build_template = forbidden
        profile.profile_template = forbidden
        shadowspill.measure_communication = forbidden
        shadowspill.measure_overlap = forbidden
    sys.argv = [module, directory]
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
