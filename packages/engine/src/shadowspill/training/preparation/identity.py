"""Portable source identities for CPU setup semantics, independent of hardware."""

from functools import lru_cache
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import shadowspill_client
from shadowspill_client.protocol.definitions import DefinitionSource


@lru_cache(maxsize=1)
def setup_sources() -> tuple[DefinitionSource, DefinitionSource]:
    """Pin Python definitions/defaults; runtime compatibility is separate.

    Hash relative paths and bytes, not Git heads or installation locations. CPU
    and CUDA wheels with the same Python sources describe the same settings.
    A running service treats its installation as immutable until restart.
    """
    sources = []
    for name, root in (
        ("shadowspill", Path(__file__).resolve().parents[2]),
        ("shadowspill-client", Path(shadowspill_client.__file__).resolve().parent),
    ):
        digest = sha256()
        for path in sorted(root.rglob("*.py")):
            digest.update(path.relative_to(root).as_posix().encode() + b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        sources.append(DefinitionSource(name, version(name), digest.hexdigest()))
    return sources[0], sources[1]
