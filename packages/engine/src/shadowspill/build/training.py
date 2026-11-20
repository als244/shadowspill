"""Build an objective and persistent gradients without optimizer computation."""

from collections.abc import Callable, Mapping, Sequence
from contextlib import closing
from dataclasses import replace
from typing import Any

import torch
from torch import nn

from shadowspill.artifacts.program import Program
from shadowspill.build.lowering.training.program import lower_training
from shadowspill.operator_preparation import OperatorPreparation
from shadowspill.operator_preparation.lifecycle import BuildPreparations
from shadowspill.parallelism import Parallelism
from shadowspill.pytorch.accelerator import resolve_device
from shadowspill.pytorch.capture.gradients.destinations import (
    GradientDestinationRule,
    normalize_gradient_destination_rules,
)
from shadowspill.pytorch.chunking import ChunkPolicy, ChunkRegions
from shadowspill.pytorch.graph_pairs.policy import GraphPairPolicy
from shadowspill.pytorch.partition import PartitionSpec
from shadowspill.resources import Resource
from shadowspill.resources.description import validate_resources

from .common import build_setup, compile_artifacts
from .finalize import finalize_tasks
from .gradient_ops import capture_gradient_operations
from .mutations import pad_mutations
from .optimizer import capture_update
from .optimizer_alignment import optimizer_input_offsets
from .parallelism import (
    attach_parallelism,
    model_parallelism,
    validate_step,
)
from .resources import attach_resources
from .training_capture import capture_training_model


def build_forward_backward(
    model: nn.Module,
    *,
    objective: Callable[..., Any],
    example_args: Sequence[Any] = (),
    example_kwargs: Mapping[str, Any] | None = None,
    num_examples: int = 1,
    example_calls: Sequence[tuple[Sequence[Any], Mapping[str, Any]]] | None = None,
    fuse_narrow_gradient_accumulation: bool = True,
    gradient_destination_rules: Sequence[GradientDestinationRule] = (),
    partition: PartitionSpec = "auto",
    chunking: ChunkPolicy | ChunkRegions | None = None,
    graph_pair_policy: GraphPairPolicy | None = None,
    grad_dtype: torch.dtype | None = None,
    parallelism: Parallelism | None = None,
    resources: Mapping[str, Resource] | None = None,
    operator_preparation: Sequence[OperatorPreparation] = (),
    rank_map: Mapping[int, int] | None = None,
    target: str | torch.device = "cuda",
) -> Program:
    """Capture a model's objective and backward as a Program, without an optimizer.

    Gradients persist after each run, which either starts them fresh or adds
    to the stored ones; no optimizer is captured. Each stage is finalized in a
    form that saves what its backward needs and one that recomputes it;
    planning later chooses among the forms and orders the examples. The
    model's parameters may be on the CPU, the meta device or the target;
    build does not change them. Nothing is profiled, no pools are opened and
    no persistent state is initialized.

    Parameters
    ----------
    model : torch.nn.Module
        The model.
    objective : callable
        Called as ``objective(model, *args, **kwargs)`` for each example. It
        returns a scalar loss, a ``(loss, metrics)`` pair or an
        `~shadowspill.pytorch.ObjectiveResult`, and owns normalization
        across examples and ranks.
    example_args : sequence, default ()
        Positional arguments of one representative example. Runs supply
        examples of the same signature: tensor values may differ, but not
        structure, shapes, dtypes or other Python values.
    example_kwargs : mapping of str to object, optional
        Keyword arguments of that example.
    num_examples : int, default 1
        Examples per run, each with that signature.
    example_calls : sequence of tuple, optional
        ``(args, kwargs)`` pairs of examples with different signatures,
        instead of ``example_args``, ``example_kwargs`` and ``num_examples``.
        Examples of one signature share their captures.
    fuse_narrow_gradient_accumulation : bool, default True
        Fuse supported gradient production into its accumulation
        destination.
    gradient_destination_rules : sequence of GradientDestinationRule, default ()
        Contracts by which opaque operators accumulate in place into gradient
        destinations, across shared uses and chunks. Outputs no rule matches
        accumulate ordinarily.
    partition : str or PartitionPolicy, default "auto"
        Stage boundaries, as in `build_forward`.
    chunking : ChunkPolicy or ChunkRegions, optional
        Execution regions and chunks within stages, as in `build_forward`. It
        may also restrict the stage forms built.
    graph_pair_policy : GraphPairPolicy, optional
        What each stage's save and recompute forms keep for the backward. By
        default ``GraphPairPolicy()``.
    grad_dtype : torch.dtype, optional
        Dtype of stored gradients. By default each parameter's dtype.
    parallelism : Parallelism, optional
        Ownership across ranks. By default one rank.
    resources : mapping of str to Resource, optional
        Shared resources the model's operators use, by name.
    operator_preparation : sequence of OperatorPreparation, default ()
        Adapters that save and restore external kernel selections, as in
        `build_forward`.
    rank_map : mapping of int to int, optional
        Launcher rank to logical rank, when ``resources`` need every worker
        during build.
    target : str or torch.device, default "cuda"
        The device to finalize code on, as in `build_forward`.

    Returns
    -------
    Program
        The forward/backward Program. Its runs return a `TrainingResult`.

    Raises
    ------
    CaptureError
        If the computation cannot be captured as a fixed graph, or the
        partition is invalid.
    ObjectiveError
        If the objective does not return a scalar differentiable loss.
    CompilationError
        If a task's code cannot be finalized.
    TypeError
        If ``graph_pair_policy`` is not a `GraphPairPolicy`.
    """
    return _build_training(
        model,
        objective=objective,
        optimizer=None,
        example_args=example_args,
        example_kwargs=example_kwargs,
        num_examples=num_examples,
        example_calls=example_calls,
        fuse_narrow_gradient_accumulation=fuse_narrow_gradient_accumulation,
        gradient_destination_rules=gradient_destination_rules,
        partition=partition,
        chunking=chunking,
        graph_pair_policy=graph_pair_policy,
        grad_dtype=grad_dtype,
        parallelism=parallelism,
        resources=resources,
        operator_preparation=operator_preparation,
        rank_map=rank_map,
        target=target,
    )


def _build_training(
    model: nn.Module,
    *,
    objective: Callable[..., Any],
    optimizer: torch.optim.Optimizer | None,
    example_args: Sequence[Any],
    example_kwargs: Mapping[str, Any] | None,
    num_examples: int,
    example_calls: Sequence[tuple[Sequence[Any], Mapping[str, Any]]] | None,
    fuse_narrow_gradient_accumulation: bool,
    gradient_destination_rules: Sequence[GradientDestinationRule],
    partition: PartitionSpec,
    graph_pair_policy: GraphPairPolicy | None,
    grad_dtype: torch.dtype | None,
    target: str | torch.device,
    parallelism: Parallelism | None = None,
    master_dtype: torch.dtype | None = None,
    runtime_gradient_scale: bool = False,
    chunking: ChunkPolicy | ChunkRegions | None = None,
    resources: Mapping[str, Resource] | None = None,
    operator_preparation: Sequence[OperatorPreparation] = (),
    rank_map: Mapping[int, int] | None = None,
) -> Program:
    gradient_destination_rules = normalize_gradient_destination_rules(
        gradient_destination_rules
    )
    if type(runtime_gradient_scale) is not bool:
        raise TypeError("runtime_gradient_scale must be a bool")
    if graph_pair_policy is not None and not isinstance(
        graph_pair_policy, GraphPairPolicy
    ):
        raise TypeError("graph_pair_policy must be a GraphPairPolicy or None")
    spec, parameters = model_parallelism(model, parallelism)
    validate_resources(resources or {}, spec)
    if optimizer is not None:
        validate_step(spec, parameters)
    device = resolve_device(target)
    assert device.index is not None
    with closing(BuildPreparations(operator_preparation, device)) as preparations:
        captured = capture_training_model(
            model,
            objective,
            example_args,
            dict(example_kwargs or {}),
            num_examples=num_examples,
            example_calls=example_calls,
            fuse_narrow_gradient_accumulation=fuse_narrow_gradient_accumulation,
            gradient_destination_rules=gradient_destination_rules,
            partition=partition,
            graph_pair_policy=graph_pair_policy,
            gradient_dtype=grad_dtype,
            device_index=device.index,
            gradient_shards={
                name: len(parameter.gradients.groups[0])
                for parameter in parameters
                if parameter.trainable and len(parameter.gradients.groups[0]) > 1
                for name in parameter.names
            },
        )
        captured = pad_mutations(captured, parameters)
        if chunking is not None:
            from .chunking import chunk_training

            captured = chunk_training(
                captured,
                chunking,
                fuse_narrow_gradient_accumulation=fuse_narrow_gradient_accumulation,
                gradient_destination_rules=gradient_destination_rules,
            )
        artifacts = [
            artifact
            for example in captured.examples
            for stage in example.stages
            for variant in stage.graph_pairs.variants
            for artifact in (variant.pair.forward, variant.pair.backward)
        ]
        gradient_operations = None
        if spec.world_size > 1:
            gradient_operations = capture_gradient_operations(
                model,
                captured,
                parameters,
                grad_dtype,
                destination="gradients" if optimizer is None else "optimizer",
            )
            for operation in gradient_operations.values():
                if operation.padding is not None:
                    artifacts.append(operation.padding)
        update = (
            None
            if optimizer is None
            else capture_update(
                model,
                optimizer,
                captured,
                gradient_dtype=grad_dtype,
                master_dtype=master_dtype,
                parameter_ownership=parameters if spec.world_size > 1 else (),
                runtime_gradient_scale=runtime_gradient_scale,
            )
        )
        if update is not None:
            from shadowspill.pytorch.capture.artifacts import GraphArtifact

            for task in update.captured.update_tasks:
                assert isinstance(task.artifact, GraphArtifact)
                artifacts.append(task.artifact)
        compiled = compile_artifacts(
            artifacts,
            device,
            input_offsets=optimizer_input_offsets(update, parameters),
        )
        program = lower_training(
            captured,
            compiled,
            setup=build_setup(device),
            gradient_dtype=grad_dtype,
            update=update,
            gradient_operations=gradient_operations,
            parallelism=spec,
            parameter_ownership=parameters if spec.world_size > 1 else (),
        )
        program = replace(
            program,
            bindings={**program.bindings, "compiled_device_index": device.index},
        )
        program = attach_resources(
            attach_parallelism(program, spec, parameters), resources or {}
        )
        return finalize_tasks(
            program, device, rank_map=rank_map, preparations=preparations
        )
