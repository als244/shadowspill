"""Resume an interrupted build without replacing completed Programs."""

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from shadowspill import Program
from shadowspill.orchestration.launch.workers import Workers

from .target import check_target


def completed(path: Path, group: Workers, *, kind: str, num_examples: int = 1) -> bool:
    # Every participant must take the same branch before a communicating build.
    if not all(group.gather(path.is_file())):
        return False
    error = None
    try:
        program = Program.load(path)
        check_target(program)
        if program.bindings.get("kind") != kind:
            raise ValueError("completed Program has a different execution kind")
        if program.bindings.get("num_examples", 1) != num_examples:
            raise ValueError("completed Program has a different example count")
    except Exception as failure:
        error = str(failure)
    errors = [value for value in group.gather(error) if value is not None]
    if errors:
        raise ValueError(f"cannot reuse completed build {path}: {'; '.join(errors)}")
    print(f"reusing completed build {path}", flush=True)
    return True


def check_metadata(path: Path, metadata: Mapping[str, Any], group: Workers) -> None:
    error = None
    try:
        if (
            group.rank == 0
            and path.is_file()
            and json.loads(path.read_text()) != json.loads(json.dumps(metadata))
        ):
            raise ValueError("adapter metadata differs from the completed build")
    except Exception as failure:
        error = str(failure)
    errors = [value for value in group.gather(error) if value is not None]
    if errors:
        raise ValueError(f"cannot reuse build metadata {path}: {'; '.join(errors)}")


def save_metadata(path: Path, metadata: Mapping[str, Any]) -> None:
    """A completed Program can recover missing metadata from its frozen adapter."""
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
            temporary = Path(f.name)
            json.dump(metadata, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
