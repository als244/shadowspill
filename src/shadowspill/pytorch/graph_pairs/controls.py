"""Resolve authentic saved controls within one AOT graph pair."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from shadowspill.pytorch.capture.artifacts import AotGraphPair

from .artifacts import (
    DifferentiatedStage,
    GraphPairVariant,
    PartitionedTrainingCapture,
    TaskGraphPairs,
)


def resolve_partitioned_saved_controls(
    captures: tuple[PartitionedTrainingCapture, ...],
    resolve_pair: Callable[[AotGraphPair, str | None], AotGraphPair],
    metadata_digests: tuple[str | None, ...] | None = None,
) -> tuple[PartitionedTrainingCapture, ...]:
    """Bind producer-derived saved controls to every backward occurrence.

    ``metadata_digests`` aligns with ``captures``: producer executions are
    shared per (producer contract, declared profiling metadata), matching
    profile identity, so structurally identical microbatches reuse one
    saved-control production while metadata-distinguished microbatches
    keep their own.
    """

    digests = metadata_digests or (None,) * len(captures)
    return tuple(
        replace(
            capture,
            stages=tuple(
                _resolve_stage(stage, resolve_pair, metadata_digest)
                for stage in capture.stages
            ),
        )
        for capture, metadata_digest in zip(captures, digests, strict=True)
    )


def _resolve_stage(
    stage: DifferentiatedStage,
    resolve_pair: Callable[[AotGraphPair, str | None], AotGraphPair],
    metadata_digest: str | None,
) -> DifferentiatedStage:
    graph_pairs = stage.graph_pairs
    return replace(
        stage,
        graph_pairs=TaskGraphPairs(
            structural_contract=graph_pairs.structural_contract,
            root_output_indices=graph_pairs.root_output_indices,
            variants=tuple(
                GraphPairVariant(
                    variant.option_id,
                    variant.memory_budget,
                    resolve_pair(variant.pair, metadata_digest),
                    variant.accumulates,
                )
                for variant in graph_pairs.variants
            ),
            reference_option_id=graph_pairs.reference_option_id,
        ),
    )


__all__ = ["resolve_partitioned_saved_controls"]
