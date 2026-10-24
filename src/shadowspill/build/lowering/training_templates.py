"""Resolve every stage form once; traversal and gradient writers stay unselected."""

from dataclasses import asdict
from typing import Any

from shadowspill.pytorch.graph_pairs import PartitionedTrainingCapture
from shadowspill.pytorch.lowering.training.artifacts import (
    PreparedStageVariant,
    TrainingBoundaries,
    TrainingObjects,
)
from shadowspill.pytorch.lowering.training.bindings import _prepare_stage_variant
from shadowspill.task.storage import StorageRootKind

from ..metadata import module_paths
from .catalog import CompiledTasks


def stage_templates(
    captures: tuple[PartitionedTrainingCapture, ...],
    objects: TrainingObjects,
    boundaries: TrainingBoundaries,
    compiled: CompiledTasks,
) -> list[list[list[dict[str, Any]]]]:
    result = []
    parameter_ids = set(objects.parameter_objects.values())
    for position, capture in enumerate(captures):
        stages = []
        for stage_index, stage in enumerate(capture.stages):
            paths = module_paths(stage.example.stage.graph_module)
            forms = []
            for option in stage.graph_pairs.variants:
                prepared = _prepare_stage_variant(
                    position,
                    stage_index,
                    stage,
                    option,
                    objects,
                    boundaries,
                    compiled,
                    None,
                    parameter_ids,
                    boundaries.object_ids[position][stage_index],
                    terminal=stage_index == len(capture.stages) - 1,
                )
                forms.append(
                    _form(
                        prepared,
                        option.option_id,
                        option.accumulates,
                        objects,
                        compiled,
                        paths,
                    )
                )
            stages.append(forms)
        result.append(stages)
    return result


def _form(
    item: PreparedStageVariant,
    variant: str,
    accumulates: bool,
    objects: TrainingObjects,
    compiled: CompiledTasks,
    paths: tuple[str, ...],
) -> dict[str, Any]:
    pair = item.pair
    contract = compiled.contract(pair.backward)
    roots = {view.leaf_index: view.root_id for view in contract.output_views}
    retained = {roots[slot.leaf_index] for slot in item.contributions}
    return {
        "variant": variant,
        "accumulates": accumulates,
        "saved_aliases": list(
            dict.fromkeys(
                objects.catalog.alias_id(value)
                for value in item.saved_internal_object_ids
            )
        ),
        "forward": {
            "profile_key": compiled.key(pair.forward, paths),
            "inputs": [asdict(slot) for slot in item.forward_inputs],
            "outputs": [asdict(slot) for slot in item.forward_outputs],
            "mutations": item.mutation_object_ids,
            "replacements": item.replacement_output_leaves,
            "handoffs": [asdict(slot) for slot in item.forward_storage_handoffs],
            "public_outputs": item.public_output_leaves,
        },
        "backward": {
            "profile_key": compiled.key(pair.backward, paths),
            "inputs": [asdict(slot) for slot in item.backward_inputs],
            "contributions": [asdict(slot) for slot in item.contributions],
            "handoffs": [asdict(slot) for slot in item.backward_storage_handoffs],
            "replacements": tuple(
                mutation.replacement_output_leaf
                for mutation in contract.mutations
                if mutation.replacement_output_leaf is not None
                and contract.roots[roots[mutation.replacement_output_leaf]].kind
                is not StorageRootKind.INPUT
            ),
            "discarded": tuple(
                view.leaf_index
                for view in contract.output_views
                if view.root_id not in retained
            ),
        },
    }
