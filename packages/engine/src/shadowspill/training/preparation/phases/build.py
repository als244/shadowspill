"""Run the build phase of a resolved public training preparation job."""

from shadowspill.training.preparation.phases.operations import build as run

if __name__ == "__main__":
    import sys

    run(sys.argv[1])
