"""Build a complete training update; scheduling remains in planning."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch import nn

from shadowspill.artifacts.program import Program
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
    parallelism: Parallelism | None = None,
    resources: Mapping[str, Resource] | None = None,
    rank_map: Mapping[int, int] | None = None,
    target: str | torch.device = "cuda",
) -> Program:
    """Capture gradients and local updates without initializing persistent state.

    The supplied optimizer must refer to the model's parameters. Neither is
    changed. Saved defaults describe lazily created optimizer state; actual
    tensor contents are supplied at execution. CPU planning chooses when each
    independent update runs after its required gradient writes and weight reads.
    ``master_dtype`` optionally captures separate optimizer parameters; model
    compute precision is unchanged. ``grad_dtype`` controls stored gradients
    independently. Optimizer state dtypes come from that optimizer's own rules.
    Trainable tensor wrappers require an explicit ``master_dtype`` even if their
    logical dtype matches it. The bounded initializer must fill their physical
    components and ``master.<parameter_name>`` separately.
    ``example_args``/``example_kwargs`` plus ``num_examples`` describe repeated
    geometry. Optional ``example_calls=[(args, kwargs), ...]`` describes distinct
    geometries instead; planning still chooses their execution order.
    ``graph_pair_policy`` defines independent saved-state cuts for the ``save``
    and ``recompute`` graphs; ``None`` selects ``GraphPairPolicy()``. ``chunking``
    may restrict allowed stage variants; planning selects among measured
    alternatives and chooses residency. A different policy requires building
    and profiling its resulting implementations before planning.
    ``gradient_destination_rules`` supplies versioned opaque-operator contracts
    for grouped in-place gradient accumulation across shared uses and chunks.
    It is independent of retention and chunk selection; unmatched outputs keep
    ordinary accumulation and all selected implementations are still profiled.
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
        parallelism=parallelism,
        resources=resources,
        rank_map=rank_map,
        target=target,
    )
