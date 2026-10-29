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


def tied(embedding_key: str, head_key: str) -> Source:
    """The embedding of a checkpoint that stores its tied head beside it. The
    stored head must equal the embedding, checked over whatever is read: the
    whole tensors when the parameter is assembled, the range alone when a
    chunk is copied. The shape pass reads no values and checks nothing."""

    def check(embedding: torch.Tensor, head: torch.Tensor) -> None:
        if not torch.equal(embedding, head):
            raise ValueError(
                "the checkpoint ties its head to its embedding but stores a head"
                " that differs from it"
            )

    def assemble(values: list[torch.Tensor]) -> torch.Tensor:
        embedding, head = values
        if not embedding.is_meta:
            check(embedding, head)
        return embedding

    def copy_range(reader, offset: int, destination: torch.Tensor) -> None:
        stop = offset + destination.numel()
        embedding = reader.tensor(embedding_key).view(-1)[offset:stop]
        check(embedding, reader.tensor(head_key).view(-1)[offset:stop])
        destination.copy_(embedding)

    return Source((embedding_key, head_key), assemble, copy_range)


class Qwen3Checkpoint(Checkpoint):
    config_from_metadata = staticmethod(from_huggingface)

    def sources(self, name: str, config) -> Source:
        if name == "embed.weight":
            if config.tied_embeddings and "lm_head.weight" in self.reader.info:
                return tied("model.embed_tokens.weight", "lm_head.weight")
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
