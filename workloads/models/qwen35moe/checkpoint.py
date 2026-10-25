"""Qwen 3.5 MoE weights from a Hugging Face checkpoint.

The query/gate projection keeps Hugging Face's per-head interleaving, as
the MoE attention implementation expects. Other attention and linear-attention
parameters load as the dense Qwen 3.5 family's do. The routed experts are the
checkpoint's fused banks,
``mlp.experts.gate_up_proj`` of shape ``[E, 2I, H]`` with each expert's
gate rows before its up rows and ``mlp.experts.down_proj`` of ``[E, H,
I]``, which are this model's ``w13`` and ``w2`` as they are; the router is
``mlp.gate``; the shared expert's gate and up projections join as its
``gate_up``, its down projection and its own gate load as they are. The
vision tower and the multi-token-prediction head are ignored.
"""

from __future__ import annotations

from workloads.models.qwen35.checkpoint import PREFIX, block_source
from workloads.models.shared.checkpoint import (
    Checkpoint,
    Source,
    concatenated,
    layer_name,
    tensor,
)

from .config import from_huggingface

MIXTURE = {
    "moe.router.weight": "mlp.gate.weight",
    "moe.w13": "mlp.experts.gate_up_proj",
    "moe.w2": "mlp.experts.down_proj",
    "moe.shared.down.weight": "mlp.shared_expert.down_proj.weight",
    "moe.shared.gate.weight": "mlp.shared_expert_gate.weight",
}


class Qwen35MoECheckpoint(Checkpoint):
    config_from_metadata = staticmethod(from_huggingface)
    ignored_prefixes = ("model.visual.", "mtp.")

    def sources(self, name: str, config) -> Source:
        if name == "embed.weight":
            return tensor(PREFIX + "embed_tokens.weight")
        if name == "final_norm.weight":
            return tensor(PREFIX + "norm.weight")
        if name == "lm_head.weight":
            return tensor("lm_head.weight")
        block = layer_name(name)
        if block is not None:
            layer, rest = block
            prefix = f"{PREFIX}layers.{layer}."
            if rest == "mixer.wq.weight" and config.layer_kind(layer) == "full":
                return tensor(prefix + "self_attn.q_proj.weight")
            if rest in MIXTURE:
                return tensor(prefix + MIXTURE[rest])
            if rest == "moe.shared.gate_up.weight":
                return concatenated(
                    prefix + "mlp.shared_expert.gate_proj.weight",
                    prefix + "mlp.shared_expert.up_proj.weight",
                )
            source = block_source(prefix, rest, config, config.layer_kind(layer))
            if source is not None:
                return source
        raise ValueError(f"no checkpoint source for {name}")


__all__ = ["Qwen35MoECheckpoint"]
