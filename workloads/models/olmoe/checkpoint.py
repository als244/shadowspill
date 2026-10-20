"""OLMoE weights from a Hugging Face checkpoint.

Per block, the norms, the four attention projections and the query and key
norms (over the whole projection width) load as they are; the router is
``mlp.gate``; the expert banks are stacked from the per-expert
projections, ``w13`` as each expert's gate projection followed by its up
projection and ``w2`` as its down projection.
"""

from __future__ import annotations

from workloads.models.shared.checkpoint import (
    Checkpoint,
    Source,
    bank,
    concatenated,
    layer_name,
    tensor,
)

from .config import from_huggingface

BLOCK = {
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "attn.wq.weight": "self_attn.q_proj.weight",
    "attn.wk.weight": "self_attn.k_proj.weight",
    "attn.wv.weight": "self_attn.v_proj.weight",
    "attn.wo.weight": "self_attn.o_proj.weight",
    "attn.q_norm.weight": "self_attn.q_norm.weight",
    "attn.k_norm.weight": "self_attn.k_norm.weight",
    "moe.router.weight": "mlp.gate.weight",
}


def expert_banks(prefix: str, experts: int) -> dict[str, Source]:
    """``w13`` and ``w2`` banks from per-expert gate, up and down projections."""
    return {
        "w13": bank(
            [
                concatenated(
                    f"{prefix}mlp.experts.{index}.gate_proj.weight",
                    f"{prefix}mlp.experts.{index}.up_proj.weight",
                )
                for index in range(experts)
            ]
        ),
        "w2": bank(
            [
                tensor(f"{prefix}mlp.experts.{index}.down_proj.weight")
                for index in range(experts)
            ]
        ),
    }


class OLMoECheckpoint(Checkpoint):
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
            if rest == "moe.w13_experts":
                return banks["w13"]
            if rest == "moe.w2_experts":
                return banks["w2"]
        raise ValueError(f"no checkpoint source for {name}")


__all__ = ["OLMoECheckpoint", "expert_banks"]
