"""Put cross-stage gradient additions inside the captured backward."""

from dataclasses import replace

from shadowspill.pytorch.capture.gradients import accumulate_gradient_outputs
from shadowspill.pytorch.graph_pairs import PartitionedTrainingCapture
from shadowspill.task.inputs import TaskInputRole


def include_shared_gradients(
    capture: PartitionedTrainingCapture,
    *,
    fuse_narrow_gradient_accumulation: bool = True,
) -> PartitionedTrainingCapture:
    """Repeated parameter uses and residual branches add to prior gradients.

    Backward visits later stages first in every supported example ordering.
    Activation gradients are local to one example; parameter gradients can also
    accumulate across examples. Derive each complete form from its creator once,
    preserving both kinds of additions in one explicit mutation contract.
    """
    seen: set[tuple[TaskInputRole, str]] = set()
    stages = list(capture.stages)
    for index in reversed(range(len(stages))):
        stage = stages[index]
        variants = []
        names: set[tuple[TaskInputRole, str]] = set()
        creators = {
            variant.option_id: variant.pair
            for variant in stage.graph_pairs.variants
            if not variant.accumulates
        }
        for variant in stage.graph_pairs.variants:
            pair = creators[variant.option_id]
            provenance = pair.gradient_provenance or pair.forward.input_provenance
            produced = {
                v.leaf_index for v in pair.backward.storage_contract.output_views
            }
            destinations = {
                leaf: (item.role, item.source)
                for leaf, item in enumerate(provenance)
                if leaf in produced
                and item.source is not None
                and item.role in {TaskInputRole.PARAMETER, TaskInputRole.ACTIVATION}
            }
            names.update(destinations.values())
            prior = tuple(
                leaf
                for leaf, key in destinations.items()
                if key in seen
                or (variant.accumulates and key[0] is TaskInputRole.PARAMETER)
            )
            if prior:
                variant = replace(
                    variant,
                    pair=replace(
                        pair,
                        backward=accumulate_gradient_outputs(
                            pair.backward,
                            prior,
                            fuse_narrow_gradient_accumulation=fuse_narrow_gradient_accumulation,
                        ),
                    ),
                )
            variants.append(variant)
        seen.update(names)
        stages[index] = replace(
            stage, graph_pairs=replace(stage.graph_pairs, variants=tuple(variants))
        )
    return replace(capture, stages=tuple(stages))
