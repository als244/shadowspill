"""Rank-local checkpoint destinations for the heterogeneous phase runner."""

from pathlib import Path

from shadowspill_workloads.recipes.text.experiment import group


def checkpoint_directory(options, home, rank):
    """Where a rank saves its checkpoints: ``checkpoints/`` in its directory
    ``home``, or ``checkpoint_dir``, each of several ranks in its own
    ``rank-xxxxx/`` there, since every rank names its archive by the step."""
    if options.get("checkpoint_dir") is None:
        return home / "checkpoints"
    directory = Path(options["checkpoint_dir"])
    return directory / f"rank-{rank:05d}" if group.ranks(options) > 1 else directory


def resume_path(resume, directory):
    """A checkpoint path as given, or a step name (``step_00000100``) that
    every rank resolves to its own save in ``directory``, where it saves its
    checkpoints."""
    path = Path(resume)
    if len(path.parts) > 1 or path.suffix == ".ssp":
        return path
    return directory / f"{path.name}.ssp"
