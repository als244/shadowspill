"""Resolve initial model values, weights and checkpoint dependencies."""

from __future__ import annotations

from shadowspill.server.artifacts import ArtifactStore
from shadowspill.server.startup.checkpoint import CheckpointResolver
from shadowspill.server.startup.weights import ResolvedWeights, resolve_weights
from shadowspill.training.preparation.setup import ResolvedSetup
from shadowspill.training.session.checkpoint import ResolvedCheckpoint
from shadowspill.training.worker.initialization import (
    prepare_frozen_initialization,
    prepare_initialization,
    restore_staging,
)
from shadowspill_client.protocol.checkpoints import WEIGHTS_MEDIA_TYPE
from shadowspill_client.protocol.creation import TrainingCreation
from shadowspill_client.protocol.requests import ArtifactRef


def resolve_initialization(
    artifacts: ArtifactStore,
    session_id: str,
    creation: TrainingCreation,
    setup: ResolvedSetup,
) -> tuple[ResolvedCheckpoint | None, ResolvedWeights | None, list[ArtifactRef]]:
    """Resolve saved values and only the model initializers they still need."""
    execution = creation.config.execution
    checkpoint = (
        None
        if creation.checkpoint is None
        else CheckpointResolver(
            artifacts,
            session_id=session_id,
            world_size=None,
            configuration=setup.config,
        ).resolve(creation.checkpoint)
    )
    weights = []
    weights_source = None
    initial = creation.config.model.initialize
    if (
        checkpoint is None
        and creation.kind == "training"
        and initial is not None
        and initial.kind == "artifact"
    ):
        assert initial.reference is not None
        ref = artifacts.get(initial.reference)
        if ref is None:
            raise ValueError("initial weights are not in this owner's artifact storage")
        if ref.media_type == WEIGHTS_MEDIA_TYPE:
            weights_source = resolve_weights(artifacts, ref, setup.config)
        else:
            weights.append(ref)
    saved = checkpoint if checkpoint is not None else weights_source
    # Complete saved state supplies all model values. Only declared
    # frozen omissions need an installed base initializer and its source.
    if saved is None and creation.kind == "training":
        prepare_initialization(setup)
    elif saved is not None:
        base = saved.manifest.info.frozen_base
        if base is not None:
            for rank in range(setup.context.parallelism.world_size):
                initial_base = prepare_frozen_initialization(setup, base, rank)
                restore_staging(initial_base, execution.max_staging_bytes)
        if base is not None and base.artifact is not None:
            weights.append(base.artifact)
    return checkpoint, weights_source, weights
