"""Stream all measured cases into a bounded, inspectable standard ZIP archive."""

from __future__ import annotations

import hashlib
import os
import zipfile
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from shadowspill.artifacts._bounded_io import _BoundedWriter
from shadowspill.artifacts._values import encoded
from shadowspill.training.preparation.artifacts.measurement_reuse import REUSE_NAME
from shadowspill.training.preparation.phases.artifacts import (
    admitted_inventory,
)
from shadowspill.training.preparation.phases.job import JOB_NAME, load_job
from shadowspill.training.preparation.planning.requests import planning_requests
from shadowspill_client._records import integer

_STAGES = ("programs", "profiled", "calibrated", "communication", "failures")
_COPY_BYTES = 64 << 10


def _files(root: Path, max_bytes: int, max_files: int) -> tuple[list[Path], int]:
    files, used, count = [], 4096, 1  # The new export also consumes a file slot.
    for directory, children, names in os.walk(root, followlinks=False):
        for name in (*children, *names):
            path = Path(directory) / name
            if path.is_symlink() or not (path.is_dir() or path.is_file()):
                raise ValueError(
                    "preparation export requires regular files/directories"
                )
            used += 4096 + (path.stat().st_size if path.is_file() else 0)
            count += 1
            if used > max_bytes:
                raise ValueError(
                    f"preparation exceeds its export disk envelope of {max_bytes} bytes"
                )
            if count > max_files:
                raise ValueError(
                    f"preparation exceeds its export file-count envelope of {max_files}"
                )
            relative = path.relative_to(root)
            if relative in (Path(JOB_NAME), Path(REUSE_NAME)) or (
                len(relative.parts) == 2
                and relative.parts[0] in _STAGES
                and path.suffix == ".ssp"
            ):
                if not path.is_file():
                    raise ValueError("preparation artifact must be a regular file")
                files.append(path)
    return sorted(files), used


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_COPY_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def export_measurements(
    directory: Path,
    destination: Path,
    *,
    max_disk_bytes: int,
    max_files: int,
    context: Mapping[str, Any] | None = None,
) -> Path:
    """Export original phase artifacts and a manifest; never plan or initialize.

    The caller holds the job's complete disk/host reservation. Original files,
    metadata and the growing archive share that disk allowance. Copies use 64 KiB
    blocks; metadata is capped at 16 MiB. Definition hooks run only on the CPU.
    ``destination`` must be outside the measured directory and must not exist.
    """
    integer(max_disk_bytes, "preparation export disk bytes", minimum=1)
    integer(max_files, "preparation export file count", minimum=1)
    directory, destination = directory.resolve(), destination.absolute()
    if destination.is_relative_to(directory) or destination.exists():
        raise ValueError("export requires a new path outside the preparation directory")
    files, used = _files(directory, max_disk_bytes, max_files)
    root, _, original = load_job(directory)
    inventory, rejected = admitted_inventory(root, original)
    descriptions = {
        path.relative_to(root).as_posix(): {
            "size_bytes": path.stat().st_size,
            "sha256": _hash(path),
        }
        for path in files
    }
    captures: dict[str, Any] = {}
    for name in descriptions:
        path = Path(name)
        if name in (JOB_NAME, REUSE_NAME):
            continue
        if path.stem not in original.templates:
            raise ValueError(
                "export artifact is outside the requested capture inventory"
            )
        captures.setdefault(path.stem, {})[path.parts[0]] = name
    for key in inventory.templates:
        if set(captures.get(key, {})) != set(_STAGES[:-1]):
            raise ValueError("preparation export is missing a required measured case")
    manifest = encoded(
        {
            "format": "shadowspill.operator_preparation-export",
            "version": 1,
            "context": {} if context is None else dict(context),
            "configuration": original.layouts.setup.config.to_dict(),
            "planning": planning_requests(inventory),
            "files": descriptions,
            "captures": {
                key: {
                    "required": key in inventory.templates,
                    "files": captures.get(key, {}),
                }
                for key in original.templates
            },
            "rejected": [asdict(row) for row in rejected],
        }
    )
    if len(manifest) > 16 << 20:
        raise ValueError("preparation export manifest exceeds its metadata bound")
    # Exclusive creation preserves unrelated paths. Publication occurs only after
    # ZIP finalization; a failed export removes only its own incomplete file.
    with destination.open("xb") as stream:
        try:
            with (
                _BoundedWriter(stream, max_disk_bytes - used) as output,
                zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive,
            ):
                archive.writestr(zipfile.ZipInfo("manifest.json"), manifest)
                for path in files:
                    name = path.relative_to(root).as_posix()
                    digest, count = hashlib.sha256(), 0
                    with (
                        path.open("rb") as source,
                        archive.open(
                            zipfile.ZipInfo(name), "w", force_zip64=True
                        ) as target,
                    ):
                        while chunk := source.read(_COPY_BYTES):
                            target.write(chunk)
                            digest.update(chunk)
                            count += len(chunk)
                    if descriptions[name] != {
                        "size_bytes": count,
                        "sha256": digest.hexdigest(),
                    }:
                        raise ValueError("preparation files changed during export")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            destination.unlink()
            raise
    return destination
