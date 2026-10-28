"""Remove what only execution needs once a cell's verdict is recorded."""

from pathlib import Path

#: Written by the build and execute phases and read by nothing after the
#: verdict: the exact initial values and the streamed checkpoint are each the
#: size of the model, and together they are most of a cell's archive.
EXECUTION_FILES = ("inputs.pt", "checkpoint.ssp")


def remove_execution_files(directory: Path) -> None:
    """Keep the archive replannable: programs, profiles and plans stay."""

    for name in EXECUTION_FILES:
        (directory / name).unlink(missing_ok=True)
