"""Small atomic file writers, independent of caches or model preparation."""

import json
import os
import tempfile
from collections.abc import Callable, Mapping
from contextlib import suppress
from pathlib import Path
from typing import BinaryIO


def atomic_json(path: Path, value: Mapping[str, object]) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_write(path: Path, write: Callable[[BinaryIO], None]) -> None:
    """Write a file whole or not at all: `write` fills a temporary file beside
    it, which then replaces it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            write(output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        with suppress(FileNotFoundError):
            os.unlink(temporary)


def atomic_text(path: Path, value: str) -> None:
    def write(output: BinaryIO) -> None:
        output.write(value.encode("utf-8"))

    atomic_write(path, write)
