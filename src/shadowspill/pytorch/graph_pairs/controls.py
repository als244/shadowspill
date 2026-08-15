"""Resolve authentic saved controls within one AOT graph pair."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from shadowspill.pytorch.capture.artifacts import AotGraphPair

from .artifacts import (
    DifferentiatedStage,
    GraphPairPortfolio,
    GraphPairVariant,
    PartitionedTrainingCapture,
)


def resolve_partitioned_saved_controls(
    captures: tuple[PartitionedTrainingCapture, ...],
    resolve_pair: Callable[[AotGraphPair], AotGraphPair],
) -> tuple[PartitionedTrainingCapture, ...]:
    """Bind producer-derived saved controls to every backward occurrence."""

    return tuple(
        replace(
            capture,
            stages=tuple(
                _resolve_stage(stage, resolve_pair) for stage in capture.stages
            ),
        )
        for capture in captures
    )


def _resolve_stage(
    stage: DifferentiatedStage,
    resolve_pair: Callable[[AotGraphPair], AotGraphPair],
) -> DifferentiatedStage:
    portfolio = stage.graph_pairs
    return replace(
        stage,
        graph_pairs=GraphPairPortfolio(
            structural_abi=portfolio.structural_abi,
            root_output_indices=portfolio.root_output_indices,
            variants=tuple(
                GraphPairVariant(
                    variant.option_id,
                    variant.memory_budget,
                    resolve_pair(variant.pair),
                )
                for variant in portfolio.variants
            ),
            reference_option_id=portfolio.reference_option_id,
        ),
    )


__all__ = ["resolve_partitioned_saved_controls"]
