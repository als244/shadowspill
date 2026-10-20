"""Qwen 3 MoE weights from a Hugging Face checkpoint.

Per block, the norms, the four attention projections and the per-head query
and key norms load as they are; the router is ``mlp.gate``; the expert
banks are stacked from the per-expert projections as OLMoE's are, ``w13``
from each expert's gate then up projection and ``w2`` from its down
projection.
"""

from __future__ import annotations

from workloads.models.olmoe.checkpoint import expert_banks
from workloads.models.shared.checkpoint import Checkpoint, Source, layer_name, tensor

from .config import from_huggingface

BLOCK = {
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "mixer.wq.weight": "self_attn.q_proj.weight",
    "mixer.wk.weight": "self_attn.k_proj.weight",
    "mixer.wv.weight": "self_attn.v_proj.weight",
    "mixer.wo.weight": "self_attn.o_proj.weight",
    "mixer.q_norm.weight": "self_attn.q_norm.weight",
    "mixer.k_norm.weight": "self_attn.k_norm.weight",
    "moe.router.weight": "mlp.gate.weight",
}


class Qwen3MoECheckpoint(Checkpoint):
    config_from_metadata = staticmethod(from_huggingface)

    def sources(self, name: str, config) -> Source:
        if name == "embed.weight":
            return tensor("model.embed_tokens.weight")
        if name == "final_norm.weight":
            return tensor("model.norm.weight")
        if name == "lm_head.weight":
            return tensor("lm_head.weight")
        block = layer_name(name)
        if block is not None:
            layer, rest = block
            prefix = f"model.layers.{layer}."
            if rest in BLOCK:
                return tensor(prefix + BLOCK[rest])
            banks = expert_banks(prefix, config.n_experts)
            if rest == "moe.w13":
                return banks["w13"]
            if rest == "moe.w2":
                return banks["w2"]
        raise ValueError(f"no checkpoint source for {name}")


__all__ = ["Qwen3MoECheckpoint"]
