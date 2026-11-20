"""Build a complete training update; scheduling remains in planning."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch import nn

from shadowspill.artifacts.program import Program
from shadowspill.operator_preparation import OperatorPreparation
from shadowspill.parallelism import Parallelism
from shadowspill.pytorch.capture.gradients import GradientDestinationRule
from shadowspill.pytorch.chunking import ChunkPolicy, ChunkRegions
from shadowspill.pytorch.graph_pairs.policy import GraphPairPolicy
from shadowspill.pytorch.partition import PartitionSpec
from shadowspill.resources import Resource

from .training import _build_training


def build_step(
    model: nn.Module,
    *,
    objective: Callable[..., Any],
    optimizer: torch.optim.Optimizer,
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
    master_dtype: torch.dtype | None = None,
    runtime_gradient_scale: bool = False,
    parallelism: Parallelism | None = None,
    resources: Mapping[str, Resource] | None = None,
    operator_preparation: Sequence[OperatorPreparation] = (),
    rank_map: Mapping[int, int] | None = None,
    target: str | torch.device = "cuda",
) -> Program:
    """Capture a model's complete training update as a Program.

    It captures the objective, the backward and the optimizer's update.
    Planning later chooses each stage's save or recompute form, the example
    order, and when each update runs, once the gradients it needs are written
    and the old weights it changes are read. Neither the model nor the
    optimizer is changed: optimizer state is created at execution from the
    captured defaults. Nothing is profiled, no pools are opened and no
    persistent state is initialized.

    Parameters
    ----------
    model : torch.nn.Module
        The model. Its parameters may be on the CPU, the meta device or the
        target.
    objective : callable
        Called as ``objective(model, *args, **kwargs)`` for each example, as
        in `build_forward_backward`.
    optimizer : torch.optim.Optimizer
        An optimizer over the model's parameters whose update can be
        captured: value-dependent Python branches, unsupported first-use
        state creation and step hooks are refused, and a control meant to
        change between runs, such as a scheduled learning rate, must be a
        tensor. Its state dtypes follow its own rules.
    example_args : sequence, default ()
        Positional arguments of one representative example, as in
        `build_forward_backward`.
    example_kwargs : mapping of str to object, optional
        Keyword arguments of that example.
    num_examples : int, default 1
        Examples per update, each with that signature.
    example_calls : sequence of tuple, optional
        ``(args, kwargs)`` pairs of examples with different signatures,
        instead of ``example_args``, ``example_kwargs`` and ``num_examples``.
    fuse_narrow_gradient_accumulation : bool, default True
        Fuse supported gradient production into its accumulation
        destination.
    gradient_destination_rules : sequence of GradientDestinationRule, default ()
        In-place gradient accumulation contracts for opaque operators, as in
        `build_forward_backward`.
    partition : str or PartitionPolicy, default "auto"
        Stage boundaries, as in `build_forward`.
    chunking : ChunkPolicy or ChunkRegions, optional
        Execution regions and chunks within stages, as in
        `build_forward_backward`.
    graph_pair_policy : GraphPairPolicy, optional
        What each stage's save and recompute forms keep for the backward. By
        default ``GraphPairPolicy()``.
    grad_dtype : torch.dtype, optional
        Dtype of stored gradients. By default each parameter's dtype.
    master_dtype : torch.dtype, optional
        Dtype of separate optimizer master weights; the model computes in its
        own dtypes. A parameter already at that precision keeps one copy, and
        masters never narrow a dtype. Trainable tensor wrappers require it,
        and are initialized by filling their components and
        ``master.<parameter>`` separately.
    runtime_gradient_scale : bool, default False
        Add a gradient scale control, default 1, that each optimizer consumer
        applies after converting the gradient to optimizer precision. The
        Program's ``bindings["gradient_scale_control"]`` names it;
        ``hyperparams`` of `Executor.run` sets it for one run, leaving stored
        gradients unchanged.
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
        The complete-step Program. Its runs return a `TrainingResult`.

    Raises
    ------
    CaptureError
        If the computation or the optimizer's update cannot be captured as a
        fixed graph, or the partition is invalid.
    ObjectiveError
        If the objective does not return a scalar differentiable loss.
    CompilationError
        If a task's code cannot be finalized.
    TypeError
        If ``runtime_gradient_scale`` is not a bool, or ``graph_pair_policy``
        is not a `GraphPairPolicy`.
    """
    return _build_training(
        model,
        objective=objective,
        optimizer=optimizer,
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
        master_dtype=master_dtype,
        runtime_gradient_scale=runtime_gradient_scale,
        parallelism=parallelism,
        resources=resources,
        operator_preparation=operator_preparation,
        rank_map=rank_map,
        target=target,
    )
