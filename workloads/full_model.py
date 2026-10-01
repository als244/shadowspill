"""Full-model specifications, example geometries and deterministic construction."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from typing import Any, cast

import torch
import torch.nn as nn

from workloads.common import auxiliary_share, language_model_loss
from workloads.precision import TrainingDtypes
from workloads.providers import MODEL_DTYPES, ModelImplementation

_GIB = 1 << 30
_RETAINED_HEAD_SCRATCH_BYTES = 512 << 20


@dataclass(frozen=True, slots=True)
class FullModelSpec:
    """One reproducible provider/model/geometry performance request."""

    family: str
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
        return f"{self.implementation}_{self.family}"

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["model_config"] = asdict(self.model_config)
        result["tokens_per_microbatch"] = self.tokens_per_microbatch
        result["tokens_per_step"] = self.tokens_per_step
        result["dtypes"] = self.dtypes.as_dict()
        return result


def throughput_spec(family: str, implementation: ModelImplementation) -> FullModelSpec:
    """A supplied model/geometry recipe, independent of gate budgets or verdicts."""

    from workloads.pytorch import Llama3Config, OLMoEConfig, Qwen35Config

    if family == "llama3":
        config: Any = Llama3Config.throughput()
        tokens = 8_192
    elif family == "qwen35":
        config = Qwen35Config.throughput()
        tokens = 16_384
    elif family == "olmoe":
        if implementation == "pytorch":
            raise ValueError("pure-PyTorch OLMoE full-model benchmarking is deferred")
        config = OLMoEConfig.throughput()
        tokens = 32_768
    else:
        raise ValueError(f"unknown full-model family {family!r}")
    return FullModelSpec(
        family=family,
        implementation=implementation,
        sequence_length=1_024,
        sequences_per_microbatch=tokens // 1_024,
        accumulation_count=65_536 // tokens,
        model_config=config,
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
    if manifest.implementation == "pytorch":
        if manifest.family == "olmoe":
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

    import mlops

    chunk = _head_chunk_size(
        int(callable_model.config.vocab_size),
        manifest.head_scratch_bytes,
    )
    if manifest.family == "olmoe":
        hidden, auxiliary = callable_model.hidden(tokens, sequence_lengths)
        return _with_balancing(
            mlops.head_loss(
                hidden,
                callable_model.lm_head.weight,
                targets,
                chunk_size=chunk,
                reduction="sum",
            ),
            auxiliary_share(auxiliary, targets, "sum"),
            total,
        )
    hidden = callable_model.hidden(tokens, sequence_lengths)
    summed = mlops.head_loss(
        hidden,
        callable_model.lm_head.weight,
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
    head: torch.Tensor, balancing: torch.Tensor, total: float
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """A mixture of experts' microbatch objective: the head's share of the
    step's mean loss plus the balancing term's, weighted. The head's share
    alone is the metric, since the loss read across models is the head's and
    the balancing term is the router's; a metric is not differentiated."""

    share = head / total
    return share + BALANCING_COEFFICIENT * (balancing / total), {
        HEAD_LOSS_METRIC: share.detach()
    }


def _head_chunk_size(vocabulary: int, scratch_bytes: int) -> int:
    rows = scratch_bytes // (2 * vocabulary)
    return max(512, (rows // 256) * 256)


@contextlib.contextmanager
def meta_construction(dtype: torch.dtype) -> Iterator[None]:
    """Declare structure without allocating: no storage, no values.

    The dtype is chosen here rather than cast afterwards. A cast allocates,
    which is what makes a model impossible to place anywhere but where it was
    built.
    """

    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device("meta"):
            yield
    finally:
        torch.set_default_dtype(previous)


def initialize_model(model: nn.Module) -> None:
    """Let every module fill the storage it owns.

    One traversal, used by every materialisation path, so a model built into
    the spill pool draws exactly what the same model built on the host draws.
    """

    for module in model.modules():
        reset = getattr(module, "reset_parameters", None)
        if callable(reset):
            reset()


def build_model(manifest: FullModelSpec) -> nn.Module:
    """Return this manifest's model on `meta`: structure only, no storage."""

    from importlib import import_module

    models = import_module("workloads." + manifest.implementation)
    model_types = {"llama3": models.Llama3, "qwen35": models.Qwen35}
    if manifest.family == "olmoe":
        model_types["olmoe"] = models.OLMoE
    try:
        model_type = model_types[manifest.family]
    except KeyError as exc:
        raise ValueError(
            "unsupported full-model cell "
            f"{(manifest.family, manifest.implementation)!r}"
        ) from exc
    if manifest.model_dtype not in MODEL_DTYPES:
        raise ValueError(f"model_dtype must be one of {MODEL_DTYPES}")
    with meta_construction(getattr(torch, manifest.model_dtype)):
        return model_type(manifest.model_config)


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
    "throughput_spec",
]
