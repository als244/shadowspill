"""Match replacement mutations to their declared distributed storage size."""

from dataclasses import replace

from shadowspill.parallelism.ownership import ParameterOwnership
from shadowspill.pytorch.capture.padding import pad_output_storage

from .training_capture import TrainingCapture


def pad_mutations(
    captured: TrainingCapture, parameters: tuple[ParameterOwnership, ...]
) -> TrainingCapture:
    groups = {
        name: len(parameter.weights.groups[0])
        for parameter in parameters
        for name in parameter.names
        if len(parameter.weights.groups[0]) > 1
    }
    if not groups:
        return captured
    examples = []
    for example in captured.examples:
        stages = []
        for stage in example.stages:
            variants = []
            for variant in stage.graph_pairs.variants:
                forward = variant.pair.forward
                multiples = {}
                for mutation in forward.storage_contract.mutations:
                    source = forward.input_provenance[mutation.input_position].source
                    name = (source or "").removeprefix("model.")
                    leaf = mutation.replacement_output_leaf
                    if leaf is not None and name in groups:
                        multiples[leaf] = groups[name]
                forward = pad_output_storage(forward, multiples)
                variants.append(
                    replace(variant, pair=replace(variant.pair, forward=forward))
                )
            stages.append(
                replace(
                    stage,
                    graph_pairs=replace(stage.graph_pairs, variants=tuple(variants)),
                )
            )
        examples.append(replace(example, stages=tuple(stages)))
    return replace(captured, examples=tuple(examples))
