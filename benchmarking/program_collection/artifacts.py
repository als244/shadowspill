"""Atomic phase archives, with read support for existing planning corpora."""

import errno
import hashlib
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

from benchmarking._serialization import atomic_json, json_mapping, read_mapping
from shadowspill.artifacts.profiled_program import ProfiledProgram
from shadowspill.step.program import StepProgram

from .corpus import (
    ProgramCaseIdentity,
    SavedProgramCase,
    initialize_corpus,
    load_step_program,
)

_SCHEMA = "shadowspill.planning_corpus.case/v2"
_FILENAME = "profiled_program.ssp"


def save_profiled_program(
    root: Path,
    *,
    identity: ProgramCaseIdentity,
    program: ProfiledProgram,
    metadata: Mapping[str, object],
) -> SavedProgramCase:
    """Publish one self-contained phase result; retries cannot replace evidence."""
    initialize_corpus(root)
    directory = (
        root / "cases" / identity.case_name / identity.geometry_name / program.digest
    )
    directory.parent.mkdir(parents=True, exist_ok=True)
    metadata = json_mapping(metadata, "metadata")
    staging = Path(tempfile.mkdtemp(prefix=".collect-", dir=directory.parent))
    try:
        archive = staging / _FILENAME
        program.save(archive)
        checksum = _checksum(archive)
        saved = SavedProgramCase(
            directory.resolve(), identity, program.digest, checksum, _FILENAME
        )
        atomic_json(
            staging / "manifest.json",
            {
                "schema": _SCHEMA,
                "identity": identity.to_dict(),
                "metadata": metadata,
                "profiled_program": {
                    "path": _FILENAME,
                    "program_digest": program.digest,
                    "artifact_sha256": checksum,
                },
            },
        )
        try:
            os.rename(staging, directory)
        except OSError as error:
            if error.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                raise
            saved, existing = load_program(directory)
            manifest = read_mapping(directory / "manifest.json", "case manifest")
            if (
                existing.digest != program.digest
                or saved.identity != identity
                or manifest["metadata"] != metadata
            ):
                raise FileExistsError(
                    f"immutable case contains different evidence: {directory}"
                ) from error
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return saved


def load_program(path: Path) -> tuple[SavedProgramCase, ProfiledProgram | StepProgram]:
    """Validate an archive or an older JSON case without loading CUDA or a model."""
    directory = path if path.is_dir() else path.parent
    manifest = read_mapping(directory / "manifest.json", "case manifest")
    if manifest.get("schema") != _SCHEMA:
        return load_step_program(path)
    artifact = manifest["profiled_program"]
    if artifact["path"] != _FILENAME:
        raise ValueError("unsupported profiled-program filename")
    archive = directory / _FILENAME
    if _checksum(archive) != artifact["artifact_sha256"]:
        raise ValueError("saved ProfiledProgram artifact digest does not match")
    program = ProfiledProgram.load(archive)
    if program.digest != artifact["program_digest"]:
        raise ValueError("saved ProfiledProgram content digest does not match")
    return SavedProgramCase(
        directory.resolve(),
        ProgramCaseIdentity.from_value(manifest["identity"]),
        program.digest,
        artifact["artifact_sha256"],
        _FILENAME,
    ), program


def _checksum(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
