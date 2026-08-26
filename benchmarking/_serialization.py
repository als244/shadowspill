"""Small shared primitives for immutable benchmark JSON artifacts."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any


def canonicalize_json(payload: str) -> str:
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError("saved JSON artifact is invalid") from error
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def pretty_json(payload: str) -> str:
    return json.dumps(json.loads(payload), indent=2, sort_keys=True) + "\n"


def text_digest(payload: str) -> str:
    return hashlib.sha256(canonicalize_json(payload).encode()).hexdigest()


def canonical_text_digest(payload: str) -> str:
    """Digest a payload that is already canonical, without re-parsing it.

    `text_digest` canonicalises first because it is handed arbitrary JSON. A
    caller that produced the payload with canonical separators and sorted keys
    already knows the answer, and re-parsing tens of megabytes to confirm it is
    pure cost.
    """

    return hashlib.sha256(payload.encode()).hexdigest()


def json_mapping(value: Mapping[str, object], path: str) -> dict[str, object]:
    try:
        encoded = json.dumps(dict(value), sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path} must contain only JSON values") from error
    return mapping(json.loads(encoded), path)


def read_mapping(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} at {path} cannot be read") from error
    return mapping(value, name)


def existing_artifact_is_identical(
    artifact_path: Path,
    manifest_path: Path,
    *,
    payload: str,
    manifest: Mapping[str, object],
) -> bool:
    """Return true for an identical retry and reject immutable collisions."""

    artifact_exists = artifact_path.exists()
    manifest_exists = manifest_path.exists()
    if not artifact_exists and not manifest_exists:
        return False
    if artifact_exists != manifest_exists:
        raise FileExistsError(
            f"incomplete immutable benchmark artifact at {artifact_path.parent}"
        )
    existing_payload = canonicalize_json(artifact_path.read_text())
    existing_manifest = read_mapping(manifest_path, "existing manifest")
    if existing_payload != canonicalize_json(payload) or (
        canonical_json_value(existing_manifest) != canonical_json_value(manifest)
    ):
        raise FileExistsError(
            "immutable benchmark identity already contains different evidence: "
            f"{artifact_path.parent}"
        )
    return True


def commit_immutable_directory(
    directory: Path,
    *,
    artifact_name: str,
    payload: str,
    manifest: Mapping[str, object],
    child_directories: tuple[str, ...] = (),
) -> None:
    """Publish a complete immutable artifact directory with one rename."""

    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{directory.name}.staging-", dir=directory.parent)
    )
    try:
        atomic_text(staging / artifact_name, payload)
        atomic_json(staging / "manifest.json", manifest)
        for name in child_directories:
            (staging / name).mkdir()
        try:
            os.rename(staging, directory)
        except OSError as error:
            if error.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                raise
            if not existing_artifact_is_identical(
                directory / artifact_name,
                directory / "manifest.json",
                payload=payload,
                manifest=manifest,
            ):
                raise RuntimeError(
                    "immutable artifact race was not identical"
                ) from error
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def mapping(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{path}: expected an object")
    return value


def string(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{path}: expected a string")
    return value


def integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path}: expected an integer")
    return value


def atomic_json(path: Path, value: Mapping[str, object]) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def canonical_json_value(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


__all__ = [
    "atomic_json",
    "atomic_text",
    "canonical_json_value",
    "canonical_text_digest",
    "canonicalize_json",
    "commit_immutable_directory",
    "existing_artifact_is_identical",
    "integer",
    "json_mapping",
    "mapping",
    "pretty_json",
    "read_mapping",
    "string",
    "text_digest",
]
