"""Run the build phase of a resolved public training preparation job."""

from .job_phases import build as run

if __name__ == "__main__":
    import sys

    run(sys.argv[1])
