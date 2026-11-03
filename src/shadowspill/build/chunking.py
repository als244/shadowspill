"""Resolve chunk policies and bind explicit graph pairs to axis chunking."""

from __future__ import annotations

from dataclasses import replace

import torch

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.artifacts import GraphArtifact, TensorGeometry
from shadowspill.pytorch.chunking import (
    AxisChunks,
    ChunkPolicy,
    ChunkRegions,
    StageInfo,
)
from shadowspill.pytorch.chunking.geometry import Layout
from shadowspill.pytorch.chunking.inputs import resolve_input_axes
from shadowspill.pytorch.chunking.transform import chunk_artifact
from shadowspill.pytorch.partition.artifacts import StageExample
from shadowspill.task.inputs import TaskInputRole

from .capture import ForwardCapture
from .metadata import module_paths
from .training_capture import TrainingCapture


def stage_config(
    example: StageExample, policy: ChunkPolicy, *, terminal: bool
) -> AxisChunks | None:
    if not isinstance(policy, ChunkPolicy):
        raise TypeError("chunking must implement configure_stage(StageInfo)")
    stage = example.stage
    info = StageInfo(
        stage.stage_id,
        module_paths(stage.graph_module),
        tuple(
            TensorGeometry.from_tensor(x) if isinstance(x, torch.Tensor) else None
            for x in example.inputs
        ),
        tuple(x.role for x in stage.input_provenance),
        tuple(
            str(n.target)
            for n in stage.graph_module.graph.nodes
            if n.op == "call_function"
        ),
        terminal,
    )
    result = policy.configure_stage(info)
    if result is not None and not isinstance(result, AxisChunks):
        raise TypeError("configure_stage must return AxisChunks or None")
    if result is not None and terminal:
        raise CaptureError(
            "generic axis chunking excludes terminal training stages; "
            "select a preceding chunked region"
        )
    return result


def chunk_training(
    captured: TrainingCapture,
    policy: ChunkPolicy | ChunkRegions,
    *,
    fuse_narrow_gradient_accumulation: bool = True,
) -> TrainingCapture:
    if isinstance(policy, ChunkRegions):
        return _region_training(
            captured,
            policy,
            fuse_narrow_gradient_accumulation=fuse_narrow_gradient_accumulation,
        )
    examples = []
    cache: dict[tuple[str, str, tuple[Layout, ...], int], GraphArtifact] = {}
    for capture in captured.examples:
        stages = []
        for index, stage in enumerate(capture.stages):
            config = stage_config(
                stage.example, policy, terminal=index == len(capture.stages) - 1
            )
            if config is None:
                stages.append(stage)
                continue
            variants = []
            stage_id = stage.example.stage.stage_id
            selection = repr(config)
            if config.regions is not None:
                # A nested callback may distinguish structurally equal stage
                # occurrences. Compile reuse follows its resolved contracts.
                selection = f"{stage_id}:{selection}"
            for option in stage.graph_pairs.variants:
                if option.option_id not in config.variants:
                    continue
                pair = option.pair
                axes, rows = resolve_input_axes(pair.forward, config)
                key = (pair.forward.compatibility_digest, selection, axes, rows)
                forward = cache.get(key)
                if forward is None:
                    forward = chunk_artifact(
                        pair.forward,
                        axes,
                        rows,
                        config,
                        stage_id=stage_id,
                        variant=option.option_id,
                    )
                    cache[key] = forward
                else:
                    forward = replace(
                        forward,
                        example_arguments=pair.forward.example_arguments,
                        input_provenance=pair.forward.input_provenance,
                    )
                assert forward.chunking is not None
                count = pair.saved_value_count
                saved = forward.chunking.outputs[-count:] if count else ()
                tangents = tuple(
                    # A shared forward output is assembled once. Its VJP
                    # contribution must enter one child, while a summed
                    # output's cotangent broadcasts to every child.
                    "sum"
                    if forward.chunking.outputs[i] is None
                    else None
                    if forward.chunking.outputs[i] == "sum"
                    else forward.chunking.outputs[i]
                    for i in stage.differentiable_output_indices[
                        : len(stage.differentiable_output_indices)
                        - pair.specialized_unit_tangent_count
                    ]
                )
                priors = (None,) * (
                    pair.backward.argument_count - len(saved) - len(tangents)
                )
                backward_layouts = list(
                    (*(None if x == "sum" else x for x in saved), *tangents, *priors)
                )
                components = pair.forward.input_components or tuple(
                    (i, ()) for i in pair.forward.tensor_argument_positions
                )
                input_axes = {
                    logical: axis
                    for (logical, _), axis in zip(components, axes, strict=True)
                }
                for mutation in pair.backward.storage_contract.mutations:
                    leaf = mutation.replacement_output_leaf
                    if leaf is not None:
                        backward_layouts[mutation.input_position] = input_axes.get(leaf)
                backward_inputs = tuple(backward_layouts)
                if len(backward_inputs) != pair.backward.argument_count:
                    raise CaptureError(
                        f"stage {stage.example.stage.stage_id} backward chunk arity: "
                        f"saved={count}, layouts={forward.chunking.outputs}, "
                        f"tangents={stage.differentiable_output_indices}, "
                        f"arguments={pair.backward.argument_count}, "
                        f"specialized={pair.specialized_unit_tangent_count}"
                    )
                # Equal tensor geometry does not establish equal chunk domains:
                # backward ports inherit layouts from the forward retention cut.
                bkey = (
                    pair.backward.compatibility_digest,
                    selection,
                    backward_inputs,
                    rows,
                )
                backward = cache.get(bkey)
                if backward is None:
                    backward = chunk_artifact(
                        pair.backward,
                        backward_inputs,
                        rows,
                        config,
                        fuse_narrow_gradient_accumulation=fuse_narrow_gradient_accumulation,
                        stage_id=stage_id,
                        variant=option.option_id,
                    )
                    cache[bkey] = backward
                else:
                    backward = replace(
                        backward,
                        example_arguments=pair.backward.example_arguments,
                        input_provenance=pair.backward.input_provenance,
                    )
                variants.append(
                    replace(
                        option, pair=replace(pair, forward=forward, backward=backward)
                    )
                )
            if not variants:
                raise CaptureError(
                    f"stage {stage.example.stage.stage_id} has no requested "
                    "retention variant; allow its mandatory save variant"
                )
            available = {option.option_id for option in variants}
            pairs = replace(
                stage.graph_pairs,
                variants=tuple(variants),
                reference_option_id="save" if "save" in available else "recompute",
            )
            stages.append(replace(stage, graph_pairs=pairs))
        examples.append(replace(capture, stages=tuple(stages)))
    return replace(captured, examples=tuple(examples))


def chunk_forward(
    captured: ForwardCapture, policy: ChunkPolicy | ChunkRegions
) -> ForwardCapture:
    artifacts = []
    for example, artifact in zip(
        captured.partitioned.stages, captured.artifacts, strict=True
    ):
        if isinstance(policy, ChunkRegions):
            from shadowspill.pytorch.chunking.regions import region_artifact

            artifacts.append(
                region_artifact(
                    artifact, policy, stage_id=example.stage.stage_id, variant=None
                )
            )
            continue
        config = stage_config(example, policy, terminal=False)
        if config is not None:
            inputs, rows = resolve_input_axes(artifact, config)
            artifact = chunk_artifact(
                artifact, inputs, rows, config, stage_id=example.stage.stage_id
            )
        artifacts.append(artifact)
    return replace(captured, artifacts=tuple(artifacts))


def _region_training(
    captured: TrainingCapture,
    policy: ChunkRegions,
    *,
    fuse_narrow_gradient_accumulation: bool,
) -> TrainingCapture:
    from shadowspill.pytorch.chunking.regions import region_artifact

    examples = []
    for capture in captured.examples:
        stages = []
        for index, stage in enumerate(capture.stages):
            if index == len(capture.stages) - 1:
                stages.append(stage)
                continue
            variants = []
            for option in stage.graph_pairs.variants:
                if option.option_id not in policy.variants:
                    continue
                pair = option.pair
                forward = region_artifact(
                    pair.forward,
                    policy,
                    stage_id=stage.example.stage.stage_id,
                    variant=option.option_id,
                )
                assert forward.regions is not None
                count = pair.saved_value_count
                saved = forward.regions.output_domains[-count:] if count else ()
                tangent_indices = stage.differentiable_output_indices[
                    : len(stage.differentiable_output_indices)
                    - pair.specialized_unit_tangent_count
                ]
                tangents = tuple(
                    forward.regions.output_domains[i] for i in tangent_indices
                )
                inherited = [
                    *saved,
                    *tangents,
                    *(
                        {}
                        for _ in range(
                            pair.backward.argument_count - len(saved) - len(tangents)
                        )
                    ),
                ]
                components = pair.forward.input_components or tuple(
                    (i, ()) for i in pair.forward.tensor_argument_positions
                )
                logical_domains = {
                    logical: domains
                    for (logical, _), domains in zip(
                        components, forward.regions.input_domains, strict=True
                    )
                }
                for mutation in pair.backward.storage_contract.mutations:
                    if mutation.replacement_output_leaf is not None:
                        inherited[mutation.input_position] = logical_domains.get(
                            mutation.replacement_output_leaf, {}
                        )
                backward = region_artifact(
                    pair.backward,
                    policy,
                    stage_id=stage.example.stage.stage_id,
                    variant=option.option_id,
                    input_domains=tuple(inherited),
                    domain_rules=forward.regions.domain_rules,
                    fuse_narrow_gradient_accumulation=fuse_narrow_gradient_accumulation,
                    gradient_leaves=tuple(
                        logical
                        for (logical, _), provenance in zip(
                            components, pair.forward.input_provenance, strict=True
                        )
                        if provenance.role is TaskInputRole.PARAMETER
                    ),
                )
                variants.append(
                    replace(
                        option, pair=replace(pair, forward=forward, backward=backward)
                    )
                )
            if not variants:
                raise CaptureError(
                    f"checkpoint {stage.example.stage.stage_id} "
                    "has no requested retention variant"
                )
            pairs = replace(
                stage.graph_pairs,
                variants=tuple(variants),
                reference_option_id="save"
                if any(x.option_id == "save" for x in variants)
                else "recompute",
            )
            stages.append(replace(stage, graph_pairs=pairs))
        examples.append(replace(capture, stages=tuple(stages)))
    return replace(captured, examples=tuple(examples))
