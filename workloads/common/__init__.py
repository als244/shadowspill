"""Shared, provider-independent model building blocks."""

from .decoder import (
    auxiliary_share,
    GatedRMSNorm,
    Packing,
    RMSNorm,
    RotaryEmbedding,
    SequenceLengths,
    apply_rotary,
    attention_metadata,
    causal_attention,
    l2_normalize,
    language_model_loss,
    packed_metadata,
    swiglu,
)

__all__ = [
    "auxiliary_share",
    "GatedRMSNorm",
    "Packing",
    "RMSNorm",
    "RotaryEmbedding",
    "SequenceLengths",
    "apply_rotary",
    "attention_metadata",
    "causal_attention",
    "l2_normalize",
    "language_model_loss",
    "packed_metadata",
    "swiglu",
]
