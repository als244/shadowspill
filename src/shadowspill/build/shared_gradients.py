"""Put cross-stage parameter-gradient additions inside the captured backward."""

from dataclasses import replace

from shadowspill.pytorch.capture.aot import accumulate_gradient_outputs
from shadowspill.pytorch.graph_pairs import (
    PartitionedTrainingCapture,
    parameter_gradient_leaves,
)


def include_shared_gradients(
    capture: PartitionedTrainingCapture,
) -> PartitionedTrainingCapture:
    """The first example still accumulates parameters used by a later stage.

    Backward visits later stages first in every supported example ordering.
    Later-example forms already accumulate every parameter; only the creator
    form needs this selective addition. No ordering is baked into the Program.
    """
    seen: set[str] = set()
    stages = list(capture.stages)
    for index in reversed(range(len(stages))):
        stage = stages[index]
        variants = []
        names: set[str] = set()
        for variant in stage.graph_pairs.variants:
            pair = variant.pair
            provenance = pair.gradient_provenance or pair.forward.input_provenance
            parameters = {
                leaf: provenance[leaf].source
                for leaf in parameter_gradient_leaves(pair)
            }
            names.update(source for source in parameters.values() if source is not None)
            prior = tuple(leaf for leaf, source in parameters.items() if source in seen)
            if prior and not variant.accumulates:
                variant = replace(
                    variant,
                    pair=replace(
                        pair, backward=accumulate_gradient_outputs(pair.backward, prior)
                    ),
                )
            variants.append(variant)
        seen.update(names)
        stages[index] = replace(
            stage, graph_pairs=replace(stage.graph_pairs, variants=tuple(variants))
        )
    return replace(capture, stages=tuple(stages))
