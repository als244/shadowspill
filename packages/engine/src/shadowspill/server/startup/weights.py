"""Resolve an owned model-weights generation for fresh worker initialization."""

import sys
from dataclasses import dataclass

from shadowspill.server.artifacts import ArtifactStore
from shadowspill.server.startup.checkpoint import (
    read_saved_manifest,
    require_checkpoint_dependencies,
)
from shadowspill.training.worker.frozen import validate_base
from shadowspill_client.protocol.checkpoints import WEIGHTS_MEDIA_TYPE, WeightsManifest
from shadowspill_client.protocol.requests import ArtifactRef
from shadowspill_client.protocol.training import ResolvedTraining


@dataclass(frozen=True)
class ResolvedWeights:
    reference: ArtifactRef
    manifest: WeightsManifest


def resolve_weights(
    artifacts: ArtifactStore, reference: ArtifactRef, configuration: ResolvedTraining
) -> ResolvedWeights:
    """Check bounded provenance and all dependencies before conversion/startup.

    Tensor verification and target ownership use the ordinary archive converter.
    Optimizer settings and update/RNG progress are deliberately not imported.
    Installed custom initializers may still consume their own artifact formats;
    this resolver accepts only the engine's complete weights generation format.
    """
    if artifacts.get(reference.id) != reference:
        raise ValueError("initial weights are not in this owner's artifact storage")
    if reference.media_type != WEIGHTS_MEDIA_TYPE:
        raise ValueError("initial weights require a weights generation")
    with artifacts.open(reference) as stream:
        manifest = read_saved_manifest(reference, stream)
    assert isinstance(manifest, WeightsManifest)
    if manifest.info.byteorder != sys.byteorder:
        raise ValueError("initial weights byte order requires explicit conversion")
    if manifest.info.frozen_base is not None:
        validate_base(manifest.info.frozen_base, configuration)
    require_checkpoint_dependencies(artifacts, manifest)
    return ResolvedWeights(reference, manifest)
