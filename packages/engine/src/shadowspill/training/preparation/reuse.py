"""Reuse prepared artifacts while keeping budget selections and training separate."""

import json
from pathlib import Path
from typing import Any

from .candidates import available, program_paths

SHARED = ("programs", "profiles", "profiled", "communication", "overlap")


def source_settings(source: str | Path, destination: Path) -> dict[str, Any]:
    source = Path(source).resolve()
    if destination.is_relative_to(source):
        raise ValueError("run_dir must be outside prepared_from")
    options = json.loads((source / "config.json").read_text())
    if not available(source, options):
        raise ValueError("prepared_from has no successful candidates")
    for relative in program_paths(options, root=source):
        for directory in ("programs", "profiled", "communication"):
            if not (source / directory / relative).is_file():
                raise ValueError(
                    f"prepared_from needs completed measurement: {directory}/{relative}"
                )
    for candidate in available(source, options):
        if not (source / "programs" / candidate["path"] / "metadata.json").is_file():
            raise ValueError("prepared_from needs each candidate's model metadata")
    # A new budget run starts new training unless resume is explicitly supplied.
    return {
        **options,
        "run_dir": str(destination),
        "prepared_from": str(source),
        "resume": None,
    }


def link_artifacts(source: str | Path, destination: Path) -> None:
    """The child may only plan/execute; it never profiles through these links."""
    source = Path(source)
    artifacts = [
        name
        for name in (*SHARED, "candidate-failures.json")
        if (source / name).exists()
    ]
    for name in artifacts:
        link = destination / name
        if link.exists() or link.is_symlink():
            raise ValueError(f"prepared run destination already contains {name}")
    for name in artifacts:
        at = source / name
        (destination / name).symlink_to(at.resolve(), target_is_directory=at.is_dir())
