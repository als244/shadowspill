"""Shared log timestamps and JSON validation for benchmark journals."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message)
        if not message.endswith("\n"):
            handle.write("\n")
        handle.flush()


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON object {path}") from error
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{path} does not contain a JSON object")
    return value
