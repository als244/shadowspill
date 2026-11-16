"""Shared admission checks selected by each text family's registration."""


def validate_activation(text, config):
    if text.get("hidden_act", "silu") != "silu":
        raise ValueError("pretrained decoder requires the supported silu activation")


def validate_decoder(
    text, config, *, mixture=False, attention_gate=False, unclipped=False
):
    validate_activation(text, config)
    if mixture and text.get("tie_word_embeddings", False):
        raise ValueError("the supplied MoE decoders require untied embeddings")
    if text.get("rope_scaling") is not None:
        raise ValueError("the supplied decoder does not implement rope_scaling")
    if mixture and (
        text.get("mlp_only_layers") or text.get("decoder_sparse_step", 1) != 1
    ):
        raise ValueError("the supplied MoE decoder requires every layer to be sparse")
    if attention_gate and text.get("attn_output_gate", True) is not True:
        raise ValueError("the supplied Qwen3.5 decoder requires attention output gates")
    if unclipped and text.get("clip_qkv") is not None:
        raise ValueError("the supplied OLMoE decoder does not implement clip_qkv")
    layers = text.get("layer_types")
    layer_kind = getattr(config, "layer_kind", lambda index: "full")
    if layers is not None and (
        not isinstance(layers, list)
        or len(layers) != config.n_layers
        or any(
            kind != f"{layer_kind(index)}_attention"
            for index, kind in enumerate(layers)
        )
    ):
        raise ValueError("pretrained layer_types differ from the supported decoder")
