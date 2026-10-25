"""HF Q/gate rows must match the consuming attention module, not just roundtrip."""

import json
from types import SimpleNamespace

import mlops
import torch
from safetensors.torch import save_file
from torch import nn

from tests.workloads.test_checkpoints import materialized, metadata, reader


def test_loaded_query_gate_matches_huggingface_projection(tmp_path, monkeypatch):
    model = materialized("qwen35moe-tiny", max_seq_len=32).float()
    config = model.config
    attention = model.blocks[1].mixer
    generator = torch.Generator().manual_seed(12)
    weight = torch.randn(attention.wq.weight.shape, generator=generator) * 0.02
    key = "model.language_model.layers.1.self_attn.q_proj.weight"
    save_file({key: weight}, tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text(json.dumps(metadata("qwen35moe", config)))
    with reader("qwen35moe")(tmp_path) as checkpoint, torch.no_grad():
        checkpoint.copy_chunk(
            "blocks.1.mixer.wq.weight", weight.shape, 0, attention.wq.weight.view(-1)
        )
    attention.q_norm = attention.k_norm = attention.wo = nn.Identity()
    monkeypatch.setattr(mlops, "partial_rope", lambda value, *_: value)
    observed = []

    def attend(query, *_):
        observed.append(query)
        return torch.ones_like(query)

    monkeypatch.setattr(mlops, "flash_attention", attend)
    hidden = torch.randn(2, 5, config.d_model, generator=generator)
    packing = SimpleNamespace(positions=None, cu_seqlens=None, max_seqlen=5)
    actual = attention(hidden, packing, SimpleNamespace(cosine=None, sine=None))
    # Transformers splits each head's projection into its query and gate halves.
    projected = nn.functional.linear(hidden, weight)
    query, gate = projected.view(2, 5, config.n_heads, 2 * config.head_dim).chunk(
        2, dim=-1
    )
    torch.testing.assert_close(observed[0], query.reshape_as(observed[0]))
    torch.testing.assert_close(actual, gate.reshape_as(actual).sigmoid())
