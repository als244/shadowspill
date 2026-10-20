"""Qwen 3 weights from a Hugging Face checkpoint.

The names map one to one: the embedding, the final norm, the head (shared
with the embedding when tied) and, per block, the two norms, the four
attention projections, the query and key norms and the three MLP
projections. A tied checkpoint may store the head beside the embedding, as
Qwen3-1.7B's does; the stored head is then read with the embedding and must
equal it. Rotary tables are computed, not read, and the rotate-half
convention matches, so no projection is permuted.
"""

from __future__ import annotations

import torch

from workloads.models.shared.checkpoint import Checkpoint, Source, layer_name, tensor

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
    "mlp.w1.weight": "mlp.gate_proj.weight",
    "mlp.w3.weight": "mlp.up_proj.weight",
    "mlp.w2.weight": "mlp.down_proj.weight",
}


def _tied(values: list[torch.Tensor]) -> torch.Tensor:
    embedding, head = values
    if not torch.equal(embedding, head):
        raise ValueError(
            "the checkpoint ties its head to its embedding but stores a head"
            " that differs from it"
        )
    return embedding


class Qwen3Checkpoint(Checkpoint):
    config_from_metadata = staticmethod(from_huggingface)

    def sources(self, name: str, config) -> Source:
        if name == "embed.weight":
            if config.tied_embeddings and "lm_head.weight" in self.reader.info:
                return Source(("model.embed_tokens.weight", "lm_head.weight"), _tied)
            return tensor("model.embed_tokens.weight")
        if name == "final_norm.weight":
            return tensor("model.norm.weight")
        if name == "lm_head.weight":
            return tensor(
                "model.embed_tokens.weight"
                if config.tied_embeddings
                else "lm_head.weight"
            )
        block = layer_name(name)
        if block is None or block[1] not in BLOCK:
            raise ValueError(f"no checkpoint source for {name}")
        return tensor(f"model.layers.{block[0]}.{BLOCK[block[1]]}")


__all__ = ["Qwen3Checkpoint"]
