"""Qwen 3.5 weights from a Hugging Face checkpoint.

The checkpoint is the vision-language model; its text decoder's keys begin
``model.language_model.``; the vision tower and the multi-token-prediction
head are ignored. A full-attention
block's query projection carries the attention gate: Hugging Face keeps
query and gate interleaved per head, this model keeps all queries then all
gates, so the rows are reordered on load. A linear-attention block's
``w_qkvz`` is the checkpoint's ``in_proj_qkv`` followed by ``in_proj_z``,
and ``w_ba`` is ``in_proj_b`` followed by ``in_proj_a``; the convolution,
the decay and step parameters, the gated norm and the output projection
load as they are.
"""

from __future__ import annotations

from workloads.models.shared.checkpoint import (
    Checkpoint,
    Source,
    concatenated,
    deinterleaved,
    layer_name,
    tensor,
)

from .config import from_huggingface

PREFIX = "model.language_model."

FULL = {
    "mixer.wk.weight": "self_attn.k_proj.weight",
    "mixer.wv.weight": "self_attn.v_proj.weight",
    "mixer.wo.weight": "self_attn.o_proj.weight",
    "mixer.q_norm.weight": "self_attn.q_norm.weight",
    "mixer.k_norm.weight": "self_attn.k_norm.weight",
}
LINEAR = {
    "mixer.conv.weight": "linear_attn.conv1d.weight",
    "mixer.A_log": "linear_attn.A_log",
    "mixer.dt_bias": "linear_attn.dt_bias",
    "mixer.lin_norm.weight": "linear_attn.norm.weight",
    "mixer.w_out.weight": "linear_attn.out_proj.weight",
}
COMMON = {
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "mlp.w1.weight": "mlp.gate_proj.weight",
    "mlp.w3.weight": "mlp.up_proj.weight",
    "mlp.w2.weight": "mlp.down_proj.weight",
}


def block_source(prefix: str, rest: str, config, kind: str) -> Source | None:
    """A Qwen 3.5 block parameter's source, shared with the mixture family;
    ``None`` for a name this layout does not know."""

    if rest in COMMON:
        return tensor(prefix + COMMON[rest])
    if kind == "full":
        if rest == "mixer.wq.weight":
            return deinterleaved(
                prefix + "self_attn.q_proj.weight",
                config.n_heads,
            )
        if rest in FULL:
            return tensor(prefix + FULL[rest])
    else:
        if rest == "mixer.w_qkvz.weight":
            return concatenated(
                prefix + "linear_attn.in_proj_qkv.weight",
                prefix + "linear_attn.in_proj_z.weight",
            )
        if rest == "mixer.w_ba.weight":
            return concatenated(
                prefix + "linear_attn.in_proj_b.weight",
                prefix + "linear_attn.in_proj_a.weight",
            )
        if rest in LINEAR:
            return tensor(prefix + LINEAR[rest])
    return None


class Qwen35Checkpoint(Checkpoint):
    config_from_metadata = staticmethod(from_huggingface)
    ignored_prefixes = ("model.visual.", "mtp.")

    def sources(self, name: str, config) -> Source:
        if name == "embed.weight":
            return tensor(PREFIX + "embed_tokens.weight")
        if name == "final_norm.weight":
            return tensor(PREFIX + "norm.weight")
        if name == "lm_head.weight":
            return tensor(
                PREFIX + "embed_tokens.weight"
                if config.tied_embeddings
                else "lm_head.weight"
            )
        block = layer_name(name)
        if block is not None:
            layer, rest = block
            source = block_source(
                f"{PREFIX}layers.{layer}.", rest, config, config.layer_kind(layer)
            )
            if source is not None:
                return source
        raise ValueError(f"no checkpoint source for {name}")


__all__ = ["Qwen35Checkpoint", "block_source"]
