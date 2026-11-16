"""Target logprobs through row-bounded ordinary projection modules.

This path accepts arbitrary per-token cotangents, including policy objectives.
It does not use scalar-loss VJP seeds, which cannot represent those gradients.
The engine still plans retained activations and backward workspace normally.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .decoder import SequenceLengths

TARGET_LOGIT_BYTES = 64 << 20


def decoder_target_logprobs(
    self,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    sequence_lengths: SequenceLengths = None,
    *,
    head_chunk_size: int | None = None,
) -> torch.Tensor:
    """Shared model method for decoders exposing ``hidden`` and ``lm_head``."""
    return target_logprobs(
        self.hidden(tokens, sequence_lengths),
        self.lm_head,
        targets,
        chunk_size=head_chunk_size,
    )


def target_logprobs(
    hidden: torch.Tensor,
    head: nn.Linear,
    targets: torch.Tensor,
    *,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Return FP32 target logprobs with the original target shape.

    The default limits each FP32 logits block to 64 MiB or one vocabulary row,
    whichever is larger. An explicit positive row count replaces that default.
    This bounds each projection, not total retained training memory. Calling the
    ordinary head preserves installed LoRA transformations and parameter ties.
    """
    if hidden.ndim < 1 or hidden.shape[-1] != head.in_features:
        raise ValueError("hidden width differs from the language model head")
    rows = hidden.numel() // hidden.shape[-1]
    if targets.numel() != rows:
        raise ValueError("target logprobs require one target per hidden row")
    if chunk_size is not None and (type(chunk_size) is not int or chunk_size <= 0):
        raise ValueError("head chunk_size must be a positive integer")
    chunk = (
        max(1, TARGET_LOGIT_BYTES // (4 * head.out_features))
        if chunk_size is None
        else chunk_size
    )
    flat = hidden.reshape(rows, hidden.shape[-1])
    labels = targets.reshape(-1).long()
    if rows == 0:
        return -F.cross_entropy(
            head(flat).float(), labels, reduction="none"
        ).reshape_as(targets)
    values = [
        -F.cross_entropy(
            head(flat[start : start + chunk]).float(),
            labels[start : start + chunk],
            reduction="none",
        )
        for start in range(0, rows, chunk)
    ]
    return torch.cat(values).reshape_as(targets)
