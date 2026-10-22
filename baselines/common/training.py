"""What every engine trains: the catalog model with the shared objective
as its forward, so a case trains exactly what the quickstart measures, and
the helpers around its construction: the wrapper-stripping of names, the
parameters a model pins at float32, the materialization of a few by
replacement, and the re-tying of shared parameters after ``to_empty``."""

from __future__ import annotations

import re

import torch
from torch import nn

from baselines.common.geometry import Geometry
from baselines.common.models import ModelRequest


class TrainingModel(nn.Module):
    """``forward(tokens, targets)`` is one microbatch's share of the update's
    objective, the summed loss over the microbatch, with a mixture's
    balancing term, divided by this rank's tokens per update, and the
    metrics the objective reports beside it (a mixture's head loss alone;
    nothing for a dense model)."""

    def __init__(
        self, request: ModelRequest, geometry: Geometry, *, dtype: str, device="meta"
    ):
        super().__init__()
        self.model = request.build(dtype=dtype, device=device)
        self.objective = request.objective(geometry)
        self.sequence_lengths = (
            geometry.sequence_length,
        ) * geometry.sequences_per_microbatch

    def load_checkpoint(self, checkpoint, **options):
        """Write a checkpoint's values into the model this objective wraps,
        under the model's own parameter names; the reader's report."""
        return checkpoint.load_into(self.model, **options)

    def forward(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor,
        sequence_lengths: tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # A microbatch may carry its own sequence lengths, as one whose
        # geometry is searched does; otherwise it has the case's.
        lengths = (
            self.sequence_lengths if sequence_lengths is None else sequence_lengths
        )
        result = self.objective(self.model, tokens, targets, lengths)
        if isinstance(result, tuple):
            loss, metrics = result
            return loss, dict(metrics)
        return result, {}


WRAPPERS = re.compile(r"(\._checkpoint_wrapped_module|\._orig_mod)+\.")


def canonical_name(name: str) -> str:
    """The name without the checkpoint, offload and compile wrappers,
    however many of them enclose a block."""
    return WRAPPERS.sub(".", name)


def initialize(module: nn.Module) -> None:
    """Initialize the materialized shards and recompute nonpersistent buffers."""
    for child in module.modules():
        reset = getattr(child, "reset_parameters", None)
        if callable(reset):
            reset()


def pinned_parameters(request: ModelRequest) -> set[str]:
    """The names of the parameters the model keeps at float32 whatever dtype
    it is built at (a DeltaNet's decay and step parameters): built at
    bfloat16 on meta, they are the ones that are not bfloat16. Sharding
    leaves them out of its casts so they compute as the model declares."""

    model = request.build(dtype="bfloat16")
    return {
        name
        for name, parameter in model.named_parameters()
        if parameter.dtype != torch.bfloat16
    }


def materialize(module: nn.Module, names: set[str], device) -> list[nn.Parameter]:
    """Replace the named parameters with empty ones on ``device`` (a meta or
    host parameter cannot take a device tensor as its data) and return
    them, in name order."""

    result = []
    for name in sorted(names):
        owner, _, attribute = name.rpartition(".")
        holder = module.get_submodule(owner) if owner else module
        old = getattr(holder, attribute)
        new = nn.Parameter(
            torch.empty_like(old, device=device), requires_grad=old.requires_grad
        )
        setattr(holder, attribute, new)
        result.append(new)
    return result


def shared_parameter_groups(module: nn.Module) -> list[list[str]]:
    """The names that share one parameter, a tied head and its embedding,
    so a copy ``to_empty`` separates can be tied again."""

    by_identity: dict[int, list[str]] = {}
    for name, parameter in module.named_parameters(remove_duplicate=False):
        by_identity.setdefault(id(parameter), []).append(name)
    return [names for names in by_identity.values() if len(names) > 1]


def retie(module: nn.Module, groups: list[list[str]]) -> None:
    for names in groups:
        first = module.get_parameter(names[0])
        for name in names[1:]:
            owner, _, attribute = name.rpartition(".")
            setattr(module.get_submodule(owner) if owner else module, attribute, first)
