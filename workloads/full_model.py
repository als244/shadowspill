"""Shared full-model manifests and deterministic workload construction."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from typing import Any, cast

import mlops  # type: ignore[import-untyped]
import torch
import torch.nn as nn

from workloads.mlops import Llama3 as MlopsLlama3
from workloads.mlops import OLMoE as MlopsOLMoE
from workloads.mlops import Qwen35 as MlopsQwen35
from workloads.providers import (
    ModelImplementation,
    implementation_context,
)
from workloads.pytorch import Llama3 as PyTorchLlama3
from workloads.pytorch import Llama3Config, OLMoEConfig, Qwen35Config
from workloads.pytorch import Qwen35 as PyTorchQwen35

_GIB = 1 << 30
_RETAINED_HEAD_SCRATCH_BYTES = 512 << 20

#: The throughput each cell sustains today, in tokens per second. The performance
#: gate fails a cell that drops below 0.95 of its entry; a cell absent from this
#: table carries no regression gate.
#:
#: Each is the median of three consecutive matrix runs on 2026-08-22 and
#: 2026-08-23 at commit 6868613, on an idle RTX 5090 under the standard probe
#: (no checkpoint, warm step, three groups of four steps):
#:
#:     mlops_llama3    3313.4   3315.2   3322.3
#:     mlops_qwen35    2937.4   2944.5   2945.8
#:     mlops_olmoe    13808.6  13842.0  13870.9
#:
#: Run-to-run spread is under 0.5%, so the 5% margin is about ten times the
#: noise: wide enough not to fire on jitter, tight enough to catch a regression.
#: Re-measure and update these deliberately when a change is meant to move
#: throughput.
_REGRESSION_TOKENS_PER_SECOND = {
    "mlops_llama3": 3_315.2,
    "mlops_qwen35": 2_944.5,
    "mlops_olmoe": 13_842.0,
}

#: What the predecessor `dataflow` system measured on the same geometry, in
#: tokens per second. ShadowSpill replaces that system, so these are a parity
#: target rather than a regression floor: the harness reports the ratio and
#: never fails a cell on it.
#:
#: Source: `dataflow` at e04b1454, qualification runs of 2026-08-08 and
#: 2026-08-09, archived at combating_fragmentation/experiments/
#: E004-recompute-refinement/archive_INDEX.json. The geometry matches this
#: manifest exactly - sequence 1024, 65,536 tokens per step, 16 GiB execution
#: budget - and the transfer bandwidths agree within 3%, so the comparison is
#: like for like.
#:
#: ShadowSpill measures 88-90% of these as of 2026-08-23. That gap is the open
#: plan-quality item, and it is the reason these are kept: re-basing them onto
#: current numbers would erase the only standing measure of it.
_PREDECESSOR_TOKENS_PER_SECOND = {
    "mlops_llama3": 3_669.2969982952136,
    "mlops_qwen35": 3_316.344617868151,
    "mlops_olmoe": 15_654.904932252315,
}


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
    regression_tokens_per_second: float | None
    predecessor_tokens_per_second: float | None
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
    sequence_length = 1_024
    return FullModelManifest(
        family=family,
        implementation=implementation,
        sequence_length=sequence_length,
        sequences_per_microbatch=tokens // sequence_length,
        accumulation_count=65_536 // tokens,
        device_physical_capacity_bytes=16 * _GIB,
        spill_budget_bytes=112 * _GIB,
        regression_tokens_per_second=_REGRESSION_TOKENS_PER_SECOND.get(
            f"{implementation}_{family}"
        ),
        predecessor_tokens_per_second=_PREDECESSOR_TOKENS_PER_SECOND.get(
            f"{implementation}_{family}"
        ),
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
