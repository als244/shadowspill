"""The three MoE families accept leaf geometry without owning provider setup."""

import pytest
import torch

from tests.workloads import test_expert_parallel_construction as composition

architecture = composition.architecture


def test_supplied_portable_experts_need_no_live_group(architecture, monkeypatch):
    from mlops.expert_parallel import QuackMoE, QuackMoEConfig

    def forbidden(*args, **kwargs):
        raise AssertionError("portable model construction touched GPU/distributed")

    monkeypatch.setattr(torch.cuda, "current_device", forbidden)
    monkeypatch.setattr(torch.distributed, "get_world_size", forbidden)
    cls, config = architecture
    options = QuackMoEConfig(
        ep_size=2,
        num_experts=config.n_experts,
        top_k=config.top_k,
        model_dim=config.d_model,
        expert_hidden_dim=config.d_ff_expert,
        tokens_per_rank=32,
    )
    names = iter(f"test.layer.{index}" for index in range(config.n_layers))
    model = cls(
        config,
        parameter_device="meta",
        dtype=torch.bfloat16,
        expert_factory=lambda: QuackMoE.from_resource(options, next(names)),
    )
    assert all(p.is_meta for p in model.parameters())
    assert len(list(model.expert_parameters())) == 2 * config.n_layers
    assert len({block.moe.experts._handle for block in model.blocks}) == config.n_layers
    model.close()
    with pytest.raises(ValueError, match="live EP resources"):
        cls(config, ep_group=object(), expert_factory=lambda: None)
