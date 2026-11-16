"""Portable source identities for CPU setup semantics, independent of hardware."""

from dataclasses import replace
from functools import lru_cache
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import shadowspill_client
from shadowspill_client.protocol.definitions import DefinitionSource
from shadowspill_client.protocol.training import ResolvedTraining


def without_initialization(config: ResolvedTraining) -> ResolvedTraining:
    """Compare computation settings independently of weight provenance/origins.

    Every compute source, control, capability and memory policy remains exact.
    Installed hooks and captures must still be rechecked before executable reuse.
    """
    return replace(
        config,
        model=replace(config.model, initialize=None),
        reference_definition=None,
        definition_sources={
            key: value
            for key, value in config.definition_sources.items()
            if key != "reference"
        },
        origins={},
        preparation=replace(config.preparation, origins={}),
    )


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
