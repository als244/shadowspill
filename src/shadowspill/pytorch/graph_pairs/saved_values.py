"""Register forward producers without retaining their activation values."""

from __future__ import annotations

from collections.abc import Callable

from shadowspill.pytorch.capture.artifacts import AotGraphPair

from .artifacts import PartitionedTrainingCapture


def register_saved_value_producers(
    captures: tuple[PartitionedTrainingCapture, ...],
    register_pair: Callable[[AotGraphPair], None],
) -> None:
    """Retain each occurrence's producer recipe; run it only when needed.

    Values and declared metadata can differ between structurally identical
    occurrences, so registration preserves the actual graph-pair objects.
    Profile deduplication still chooses which occurrences need measuring.
    """
    for capture in captures:
        for stage in capture.stages:
            for variant in stage.graph_pairs.variants:
                register_pair(variant.pair)


__all__ = ["register_saved_value_producers"]
