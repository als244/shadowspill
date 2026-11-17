"""Identify the vocabulary and special-token policy used by a text corpus."""

import hashlib
from pathlib import Path


def tokenizer_identity(directory: str | Path) -> dict[str, str]:
    """Hash the local tokenizer files without loading any model weights."""
    directory = Path(directory)
    return {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in ("tokenizer.json", "tokenizer_config.json")
    }
