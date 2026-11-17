"""Deployment-owned checkpoint lookup before installed metadata resolution."""

import re
from dataclasses import replace
from pathlib import Path

from shadowspill_client.protocol.setup import TrainingSetup
from shadowspill_client.protocol.training import ResolvedTraining


def local_model_source(
    settings: TrainingSetup | ResolvedTraining, root: Path | None
) -> TrainingSetup | ResolvedTraining:
    """Resolve unpinned names read-only; saved setups and explicit pins stay exact.

    A deployment may use ``root/organization/model`` or ``root/model``. The
    installed local-checkpoint resolver still validates format and architecture,
    streams content checksums, and pins every file before initialization.
    """
    if root is None or not isinstance(settings, TrainingSetup):
        return settings
    weights = settings.model.initialize
    if (
        weights is None
        or weights.kind != "pretrained"
        or weights.revision is not None
        or weights.metadata
        or weights.reference is None
        or not re.fullmatch(r"[\w.-]+(?:/[\w.-]+)?", weights.reference)
        or any(part in (".", "..") for part in weights.reference.split("/"))
    ):
        return settings
    paths = {root / weights.reference, root / weights.reference.rsplit("/", 1)[-1]}
    found = {path.resolve(strict=True) for path in paths if path.exists()}
    if len(found) > 1:
        raise ValueError(f"ambiguous checkpoint directories for {weights.reference!r}")
    if not found:
        return settings
    directory = found.pop()
    if not directory.is_dir():
        raise ValueError(f"checkpoint source is not a directory: {directory}")
    return replace(
        settings,
        model=replace(
            settings.model, initialize=replace(weights, reference=str(directory))
        ),
    )
