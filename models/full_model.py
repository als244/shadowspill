"""Shared full-model manifests and deterministic workload construction."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from typing import Any, cast

import mlops  # type: ignore[import-untyped]
import torch
import torch.nn as nn

from models.mlops import Llama3 as MlopsLlama3
from models.mlops import OLMoE as MlopsOLMoE
from models.mlops import Qwen35 as MlopsQwen35
from models.providers import (
    ModelImplementation,
    implementation_context,
)
from models.pytorch import Llama3 as PyTorchLlama3
from models.pytorch import Llama3Config, OLMoEConfig, Qwen35Config
from models.pytorch import Qwen35 as PyTorchQwen35

_GIB = 1 << 30
_RETAINED_HEAD_SCRATCH_BYTES = 512 << 20


@dataclass(frozen=True, slots=True)
class FullModelManifest:
    """One reproducible provider/model/geometry performance request."""

    family: str
    implementation: ModelImplementation
    sequence_length: int
    sequences_per_microbatch: int
    accumulation_count: int
    device_physical_capacity_bytes: int
    spill_budget_bytes: int
    historical_tokens_per_second: float | None
    model_config: Any
    head_scratch_bytes: int = _RETAINED_HEAD_SCRATCH_BYTES

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
        return result


@dataclass(frozen=True, slots=True)
class FullModelCase:
    """Initialized CPU model and one complete accumulated-step template."""

    manifest: FullModelManifest
    model: nn.Module
    microbatches: tuple[tuple[object, ...], ...]

    def implementations(self) -> AbstractContextManager[Any]:
        return implementation_context(
            self.manifest.family,
            self.manifest.implementation,
        )

    def objective(self, model: nn.Module, *values: object) -> torch.Tensor:
        tokens, targets, sequence_lengths = values
        if not isinstance(tokens, torch.Tensor) or not isinstance(
            targets, torch.Tensor
        ):
            raise TypeError("performance tokens and targets must be tensors")
        callable_model: Any = model
        if self.manifest.implementation == "pytorch":
            if self.manifest.family == "olmoe":
                return cast(
                    torch.Tensor,
                    callable_model.loss(
                        tokens,
                        targets,
                        seq_lens=sequence_lengths,
                        aux_coef=0.01,
                    ),
                )
            return cast(
                torch.Tensor,
                callable_model.loss(tokens, targets, seq_lens=sequence_lengths),
            )

        chunk = _head_chunk_size(
            int(callable_model.config.vocab_size),
            self.manifest.head_scratch_bytes,
        )
        if self.manifest.family == "olmoe":
            hidden, auxiliary = callable_model.hidden(tokens, sequence_lengths)
            objective = mlops.head_loss(
                hidden,
                callable_model.lm_head.weight,
                targets,
                chunk_size=chunk,
            )
            return cast(torch.Tensor, objective + 0.01 * auxiliary)
        hidden = callable_model.hidden(tokens, sequence_lengths)
        return cast(
            torch.Tensor,
            mlops.head_loss(
                hidden,
                callable_model.lm_head.weight,
                targets,
                chunk_size=chunk,
            ),
        )

    @staticmethod
    def optimizer(parameters: Iterator[nn.Parameter]) -> torch.optim.Optimizer:
        values = list(parameters)
        groups = (
            {
                "params": [item for item in values if item.ndim >= 2],
                "weight_decay": 0.1,
            },
            {
                "params": [item for item in values if item.ndim < 2],
                "weight_decay": 0.0,
            },
        )
        return cast(torch.optim.Optimizer, mlops.optim.AdamW(groups, lr=3e-4))


def _head_chunk_size(vocabulary: int, scratch_bytes: int) -> int:
    rows = scratch_bytes // (2 * vocabulary)
    return max(512, (rows // 256) * 256)


def _manifest(
    family: str,
    implementation: ModelImplementation,
) -> FullModelManifest:
    if family == "llama3":
        config: Any = Llama3Config.throughput()
        tokens = 8_192
        historical = 3_669.2969982952136 if implementation == "mlops" else None
    elif family == "qwen35":
        config = Qwen35Config.throughput()
        tokens = 16_384
        historical = 3_316.344617868151 if implementation == "mlops" else None
    elif family == "olmoe":
        if implementation == "pytorch":
            raise ValueError("pure-PyTorch OLMoE full-model benchmarking is deferred")
        config = OLMoEConfig.throughput()
        tokens = 32_768
        historical = 15_654.904932252315
    else:
        raise ValueError(f"unknown full-model family {family!r}")
    sequence_length = 1_024
    return FullModelManifest(
        family=family,
        implementation=implementation,
        sequence_length=sequence_length,
        sequences_per_microbatch=tokens // sequence_length,
        accumulation_count=65_536 // tokens,
        device_physical_capacity_bytes=16 * _GIB,
        spill_budget_bytes=112 * _GIB,
        historical_tokens_per_second=historical,
        model_config=config,
    )


def manifests() -> tuple[FullModelManifest, ...]:
    """Return the five accepted provider cells in stable execution order."""

    return (
        _manifest("llama3", "mlops"),
        _manifest("qwen35", "mlops"),
        _manifest("olmoe", "mlops"),
        _manifest("llama3", "pytorch"),
        _manifest("qwen35", "pytorch"),
    )


def build_case(manifest: FullModelManifest, *, seed: int) -> FullModelCase:
    """Build one CPU-resident model and deterministic packed microbatches."""

    torch.manual_seed(seed)
    model_types: dict[tuple[str, ModelImplementation], type[nn.Module]] = {
        ("llama3", "pytorch"): PyTorchLlama3,
        ("llama3", "mlops"): MlopsLlama3,
        ("qwen35", "pytorch"): PyTorchQwen35,
        ("qwen35", "mlops"): MlopsQwen35,
        ("olmoe", "mlops"): MlopsOLMoE,
    }
    try:
        model_type = model_types[(manifest.family, manifest.implementation)]
    except KeyError as exc:
        raise ValueError(
            "unsupported full-model cell "
            f"{(manifest.family, manifest.implementation)!r}"
        ) from exc
    model = model_type(manifest.model_config).to(torch.bfloat16)
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


def manifest_for(family: str, implementation: ModelImplementation) -> FullModelManifest:
    for item in manifests():
        if item.family == family and item.implementation == implementation:
            return item
    raise ValueError(f"unknown full-model cell {(family, implementation)!r}")


__all__ = [
    "FullModelCase",
    "FullModelManifest",
    "build_case",
    "manifest_for",
    "manifests",
]
