"""Run the communicate phase of a resolved public training preparation job."""

from .job_phases import communicate as run

if __name__ == "__main__":
    import sys

    run(sys.argv[1])
