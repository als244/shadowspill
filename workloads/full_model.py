"""Full-model specifications, example geometries and deterministic construction."""

from __future__ import annotations

import contextlib
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from typing import Any, cast

import torch
import torch.nn as nn

from workloads.models import catalog
from workloads.models.construction import build_on_meta
from workloads.models.construction import initialize as initialize_model
from workloads.models.shared.decoder import auxiliary_share, language_model_loss
from workloads.precision import TrainingDtypes
from workloads.providers import MODEL_DTYPES, ModelImplementation

_RETAINED_HEAD_SCRATCH_BYTES = 512 << 20


@dataclass(frozen=True, slots=True)
class FullModelSpec:
    """One reproducible model/implementation/geometry performance request."""

    model: str
    implementation: ModelImplementation
    sequence_length: int
    sequences_per_microbatch: int
    accumulation_count: int
    model_config: Any
    head_scratch_bytes: int = _RETAINED_HEAD_SCRATCH_BYTES
    model_dtype: str = "bfloat16"
    master_dtype: str = "none"
    grad_dtype: str = "bfloat16"
    opt_state_dtype: str = "bfloat16"

    @property
    def entry(self) -> catalog.Entry:
        return catalog.entry(self.model)

    @property
    def family(self) -> str:
        """The family the gates name a cell by."""
        return self.entry.family

    @property
    def dtypes(self) -> TrainingDtypes:
        return TrainingDtypes(
            self.model_dtype, self.master_dtype, self.grad_dtype, self.opt_state_dtype
        )

    @property
    def tokens_per_microbatch(self) -> int:
        return self.sequence_length * self.sequences_per_microbatch

    @property
    def tokens_per_step(self) -> int:
        return self.tokens_per_microbatch * self.accumulation_count

    @property
    def identity(self) -> str:
        """The cell the gates judge: implementation and family."""
        return f"{self.implementation}_{self.family}"

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["family"] = self.family
        result["model_config"] = asdict(self.model_config)
        result["tokens_per_microbatch"] = self.tokens_per_microbatch
        result["tokens_per_step"] = self.tokens_per_step
        result["dtypes"] = self.dtypes.as_dict()
        return result


def spec(
    model: str, implementation: ModelImplementation = "mlops", **changes: Any
) -> FullModelSpec:
    """A catalog model at its retained geometry, independent of gate budgets
    or verdicts; ``changes`` apply to the entry's configuration."""

    item = catalog.entry(model)
    if implementation not in item.implementations:
        raise ValueError(
            f"{model} has no {implementation!r} implementation; "
            f"it has {', '.join(item.implementations)}"
        )
    geometry = item.geometry
    return FullModelSpec(
        model=model,
        implementation=implementation,
        sequence_length=geometry.sequence_length,
        sequences_per_microbatch=geometry.sequences_per_microbatch,
        accumulation_count=geometry.accumulation_count,
        model_config=item.config(**changes),
    )


@dataclass(frozen=True, slots=True)
class FullModelCase:
    """Initialized CPU model and one complete accumulated-step template."""

    manifest: FullModelSpec
    model: nn.Module
    microbatches: tuple[tuple[object, ...], ...]

    def implementations(self) -> AbstractContextManager[Any]:
        """Leave implementation selection to the model's operation library."""
        return contextlib.nullcontext()

    def objective(self, model: nn.Module, *values: object) -> Any:
        return full_model_objective(self.manifest, model, *values)

    @property
    def optimizer(self) -> Any:
        return self.manifest.dtypes.optimizer()


def full_model_objective(
    manifest: FullModelSpec, model: nn.Module, *values: object
) -> Any:
    """One microbatch's contribution, normalized by this recipe's update total."""
    tokens, targets, sequence_lengths = values
    if not isinstance(tokens, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise TypeError("performance tokens and targets must be tensors")
    total = float(manifest.tokens_per_step)
    callable_model: Any = model
    mixture = manifest.entry.mixture
    if manifest.implementation == "pytorch":
        if mixture:
            hidden, auxiliary = callable_model.hidden(tokens, sequence_lengths)
            return _with_balancing(
                language_model_loss(hidden, callable_model.lm_head, targets, "sum"),
                auxiliary_share(auxiliary, targets, "sum"),
                total,
            )
        summed = callable_model.loss(
            tokens, targets, seq_lens=sequence_lengths, reduction="sum"
        )
        return cast(torch.Tensor, summed / total)

    chunk = _head_chunk_size(
        int(callable_model.config.vocab_size),
        manifest.head_scratch_bytes,
    )
    if mixture:
        hidden, auxiliary = callable_model.hidden(tokens, sequence_lengths)
        return _with_balancing(
            callable_model.lm_head.loss(
                hidden,
                targets,
                chunk_size=chunk,
                reduction="sum",
            ),
            auxiliary_share(auxiliary, targets, "sum"),
            total,
            coefficient=getattr(
                callable_model.config, "router_aux_loss_coef", BALANCING_COEFFICIENT
            ),
        )
    hidden = callable_model.hidden(tokens, sequence_lengths)
    summed = callable_model.lm_head.loss(
        hidden,
        targets,
        chunk_size=chunk,
        reduction="sum",
    )
    return cast(torch.Tensor, summed / total)


#: The weight of the router's balancing term in a mixture of experts' objective.
BALANCING_COEFFICIENT = 0.01

#: The metric an MoE objective reports beside its loss: the head's share alone.
HEAD_LOSS_METRIC = "head_loss"


def _with_balancing(
    head: torch.Tensor,
    balancing: torch.Tensor,
    total: float,
    *,
    coefficient: float = BALANCING_COEFFICIENT,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """A mixture of experts' microbatch objective: the head's share of the
    step's mean loss plus the balancing term's, weighted. The head's share
    alone is the metric, since the loss read across models is the head's and
    the balancing term is the router's; a metric is not differentiated."""

    share = head / total
    return share + coefficient * (balancing / total), {HEAD_LOSS_METRIC: share.detach()}


def _head_chunk_size(vocabulary: int, scratch_bytes: int) -> int:
    rows = scratch_bytes // (2 * vocabulary)
    return max(512, (rows // 256) * 256)


def build_model(manifest: FullModelSpec) -> nn.Module:
    """Return this manifest's model on `meta`: structure only, no storage."""

    model_type = manifest.entry.model_class(manifest.implementation)
    if manifest.model_dtype not in MODEL_DTYPES:
        raise ValueError(f"model_dtype must be one of {MODEL_DTYPES}")
    return build_on_meta(
        model_type, dtype=manifest.model_dtype, config=manifest.model_config
    )


def build_case(
    manifest: FullModelSpec,
    *,
    seed: int,
) -> FullModelCase:
    """Build initialized CPU state and deterministic packed microbatches."""

    torch.manual_seed(seed)
    model = build_model(manifest)
    model.to_empty(device="cpu")
    initialize_model(model)
    model.train()
    shape = (1, manifest.tokens_per_microbatch)
    lengths = (manifest.sequence_length,) * manifest.sequences_per_microbatch
    vocabulary = int(manifest.model_config.vocab_size)
    microbatches = tuple(
        (
            torch.randint(vocabulary, shape),
            torch.randint(vocabulary, shape),
            lengths,
        )
        for _ in range(manifest.accumulation_count)
    )
    return FullModelCase(manifest, model, microbatches)


__all__ = [
    "FullModelCase",
    "FullModelSpec",
    "build_case",
    "build_model",
    "full_model_objective",
    "initialize_model",
    "spec",
]
