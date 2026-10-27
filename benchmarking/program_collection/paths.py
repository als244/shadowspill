"""Collection directories, portable paths and exclusive controller ownership."""

from __future__ import annotations

import fcntl
import os
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path

from benchmarking._serialization import atomic_json, canonical_json_value
from benchmarking.journals import read_object

from .matrix import ProgramRequest
from .settings import CollectionConfig


@dataclass(frozen=True, slots=True)
class CollectionPaths:
    """Stable filesystem paths for one configuration digest."""

    output_root: Path
    run_directory: Path
    config_path: Path
    cases_directory: Path
    controller_log: Path
    summary_path: Path
    lock_path: Path

    @classmethod
    def initialize(cls, output_root: Path, config: CollectionConfig) -> CollectionPaths:
        root = output_root.expanduser().resolve()
        run = root / "_collections" / f"{config.name}-{config.digest[:12]}"
        result = cls(
            output_root=root,
            run_directory=run,
            config_path=run / "config.json",
            cases_directory=run / "cases",
            controller_log=run / "collection.log",
            summary_path=run / "summary.json",
            lock_path=run / "collection.lock",
        )
        result.cases_directory.mkdir(parents=True, exist_ok=True)
        payload = config.to_dict()
        if result.config_path.exists():
            existing = read_object(result.config_path)
            if canonical_json_value(existing) != canonical_json_value(payload):
                raise ValueError("collection config snapshot changed for this digest")
        else:
            atomic_json(result.config_path, payload)
        return result

    def case_directory(self, request: ProgramRequest) -> Path:
        return self.cases_directory / request.case_id

    def status_path(self, request: ProgramRequest) -> Path:
        return self.case_directory(request) / "status.json"

    def request_path(self, request: ProgramRequest) -> Path:
        return self.case_directory(request) / "request.json"

    def worker_result_path(self, request: ProgramRequest, attempt: int) -> Path:
        return self.case_directory(request) / f"worker-result-{attempt:04d}.json"

    def log_path(self, request: ProgramRequest, attempt: int) -> Path:
        return self.case_directory(request) / "logs" / f"attempt-{attempt:04d}.log"


class CollectionLock(AbstractContextManager[None]):
    """Prevent two controllers from mutating one collection journal."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._descriptor: int | None = None

    def __enter__(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self._path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            os.close(descriptor)
            raise RuntimeError(f"another collector owns {self._path.parent}") from error
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"pid={os.getpid()}\n".encode())
        self._descriptor = descriptor
        return None

    def __exit__(self, *_errors: object) -> None:
        if self._descriptor is not None:
            fcntl.flock(self._descriptor, fcntl.LOCK_UN)
            os.close(self._descriptor)
            self._descriptor = None


def portable_artifact(
    paths: CollectionPaths,
    artifact: Mapping[str, object],
) -> dict[str, object]:
    result = {str(key): value for key, value in artifact.items()}
    for key in ("directory", "program_path"):
        value = result.get(key)
        if isinstance(value, str):
            result[key] = path_reference(paths, Path(value))
    return result


def path_reference(paths: CollectionPaths, path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(paths.output_root))
    except ValueError:
        return str(resolved)
