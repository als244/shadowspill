"""Operation-provider contexts shared by qualification workloads."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Any, Literal, cast

from mlops.dispatch import use_implementations  # type: ignore[import-untyped]

ModelImplementation = Literal["pytorch", "mlops"]

_MLOPS_IMPLEMENTATIONS = {
    "llama3": {
        "embedding": "builtin.embedding.deterministic",
        "rms_norm": "builtin.rms_norm.triton",
        "rope": "builtin.rope.triton_table",
        "flash_attention": "builtin.flash_attention.aten",
        "swiglu": "builtin.swiglu.triton",
        "head_loss": "builtin.head_loss.chunked",
    },
    "qwen35": {
        "embedding": "builtin.embedding.deterministic",
        "rms_norm": "builtin.rms_norm.triton",
        "gated_rms_norm": "fla.gated_rms_norm",
        "partial_rope": "builtin.partial_rope.triton",
        "flash_attention": "builtin.flash_attention.aten",
        "causal_conv_silu": "fla.causal_conv_silu",
        "l2_norm": "fla.l2_norm",
        "linear_attention": "fla.linear_attention.gated_delta_rule",
        "swiglu": "builtin.swiglu.triton",
        "head_loss": "builtin.head_loss.chunked",
    },
    "olmoe": {
        "embedding": "builtin.embedding.deterministic",
        "rms_norm": "builtin.rms_norm.triton",
        "rope": "builtin.rope.triton_table",
        "flash_attention": "builtin.flash_attention.aten",
        "moe": "builtin.moe.grouped_gemm_composed",
        "head_loss": "builtin.head_loss.chunked",
    },
}


def implementation_context(
    family: str, implementation: ModelImplementation
) -> AbstractContextManager[Any]:
    """Select the requested model-operation provider without changing core code."""

    if implementation == "pytorch":
        return nullcontext()
    try:
        selected = _MLOPS_IMPLEMENTATIONS[family]
    except KeyError as exc:
        raise ValueError(f"unknown mlops model family {family!r}") from exc
    return cast(AbstractContextManager[Any], use_implementations(selected))


__all__ = ["ModelImplementation", "implementation_context"]
