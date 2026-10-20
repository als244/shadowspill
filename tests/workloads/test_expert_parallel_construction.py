"""Workload composition/ownership contracts; these do not emulate GPU kernels."""

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from workloads.models.olmoe.config import Config as OLMoEConfig
from workloads.models.olmoe.implementations.mlops import Model as OLMoE
from workloads.models.qwen3moe.config import Config as Qwen3MoEConfig
from workloads.models.qwen3moe.implementations.mlops import Model as Qwen3MoE
from workloads.models.qwen35moe.config import Config as Qwen35MoEConfig
from workloads.models.qwen35moe.implementations.mlops import Model as Qwen35MoE


@pytest.fixture(
    params=[OLMoE, Qwen3MoE, Qwen35MoE], ids=["olmoe", "qwen3moe", "qwen35moe"]
)
def architecture(request):
    cls = request.param
    if cls is OLMoE:
        config = OLMoEConfig(3, 128, 4, 2, 32, 4, 2, 128, 97, max_seq_len=16)
    else:
        base = Qwen3MoEConfig() if cls is Qwen3MoE else Qwen35MoEConfig()
        config = replace(
            base,
            n_layers=3,
            d_model=128,
            n_heads=4,
            n_kv_heads=2,
            head_dim=32,
            n_experts=4,
            top_k=2,
            d_ff_expert=128,
            d_ff_shared=128 if base.d_ff_shared else 0,
            vocab_size=97,
            max_seq_len=16,
            lin_k_heads=2,
            lin_v_heads=4,
            lin_k_head_dim=32,
            lin_v_head_dim=32,
        )
    return cls, config


@pytest.fixture
def fake_ep(monkeypatch):
    import mlops.expert_parallel as ep

    state = SimpleNamespace(buffers=[], layers=[], events=[], fail_at=None)

    class Buffer:
        def destroy(self):
            state.events.append(("destroy", self))

    class Layer(nn.Module):
        def __init__(self, options, group, *, buffer, device):
            super().__init__()
            if len(state.layers) == state.fail_at:
                raise RuntimeError("injected construction failure")
            self.config = options
            self.group, self.communication_buffer, self.device = group, buffer, device
            self.router_weight = nn.Parameter(
                torch.ones(
                    options.num_experts, options.model_dim, dtype=options.router_dtype
                ),
                requires_grad=False,
            )
            self.gate_up_weight = nn.Parameter(torch.ones(2, 128, 256))
            self.down_weight = nn.Parameter(torch.ones(2, 128, 128))
            self.original_router = self.router_weight.detach()
            self.closed = False
            state.layers.append(self)

        def close(self):
            if not self.closed:
                state.events.append(("close", self))
                self.closed = True

        def expert_parameters(self):
            yield self.gate_up_weight
            yield self.down_weight

    def create_buffer(options, tokens, group):
        buffer = Buffer()
        state.buffers.append((buffer, options, tokens, group))
        return buffer

    # Set the lazy exports directly: accessing their old attributes would load
    # optional accelerator dependencies before installing the test stand-ins.
    monkeypatch.setitem(ep.__dict__, "QuackMoE", Layer)
    monkeypatch.setitem(ep.__dict__, "create_buffer", create_buffer)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    state.Buffer = Buffer
    return state


def test_all_layers_share_one_owned_buffer(architecture, fake_ep):
    cls, config = architecture
    group = object()
    previous_dtype = torch.get_default_dtype()
    model = cls(
        config,
        ep_group=group,
        token_capacity=256,
        device="cuda:1",
        parameter_device="cpu",
    )
    assert torch.get_default_dtype() == previous_dtype
    assert len(fake_ep.buffers) == 1
    buffer, options, tokens, actual_group = fake_ep.buffers[0]
    assert tokens == 256 and actual_group is group
    assert options.share_expert_banks and options.ep_size == 2
    assert options.renormalize_topk == (cls is not OLMoE)
    assert options.router_dtype == (torch.float32 if cls is OLMoE else torch.bfloat16)
    assert len(fake_ep.layers) == config.n_layers
    assert all(layer.communication_buffer is buffer for layer in fake_ep.layers)
    assert all(layer.device == torch.device("cuda:1") for layer in fake_ep.layers)
    assert model.embed.weight.dtype == torch.bfloat16
    assert len(list(model.expert_parameters())) == 2 * config.n_layers
    for layer in fake_ep.layers:
        # Relocation must preserve freezing and detach from the original bank.
        assert not layer.router_weight.requires_grad
        assert layer.router_weight.data_ptr() != layer.original_router.data_ptr()
        torch.testing.assert_close(layer.router_weight, layer.original_router)
    model.close()
    model.close()
    assert fake_ep.events[-1] == ("destroy", buffer)
    assert sum(kind == "destroy" for kind, _ in fake_ep.events) == 1
    assert sum(kind == "close" for kind, _ in fake_ep.events) == config.n_layers


def test_borrowed_buffer_and_precision_are_shared_options(architecture, fake_ep):
    cls, config = architecture
    buffer = fake_ep.Buffer()
    model = cls(
        config,
        ep_group=object(),
        buffer=buffer,
        device="cuda:0",
        parameter_device="cpu",
        router_dtype=torch.float32,
        compute_precision="fp8_current",
        activation_transport="fp8",
        weight_grad_dtype=torch.float32,
    )
    assert not fake_ep.buffers
    for layer in fake_ep.layers:
        assert layer.config.compute_precision == "fp8_current"
        assert layer.config.activation_transport == "fp8"
        assert layer.config.weight_grad_dtype == torch.float32
        assert layer.config.router_dtype == torch.float32
        assert layer.communication_buffer is buffer
    model.close()
    assert all(kind == "close" for kind, _ in fake_ep.events)


@pytest.mark.parametrize("router_dtype", [torch.bfloat16, torch.float32])
def test_fp32_gradient_request_reaches_router(architecture, fake_ep, router_dtype):
    cls, config = architecture
    model = cls(
        config,
        ep_group=object(),
        token_capacity=256,
        device="cuda:0",
        parameter_device="cpu",
        dtype=torch.bfloat16,
        router_dtype=router_dtype,
        weight_grad_dtype=torch.float32,
    )
    assert all(layer.config.router_dtype == router_dtype for layer in fake_ep.layers)
    assert all(
        layer.config.router_weight_grad_dtype == torch.float32
        for layer in fake_ep.layers
    )
    model.close()


def test_parallel_routing_keeps_fp32_logits(monkeypatch):
    from mlops.expert_parallel.quack import router

    from workloads.models.shared.expert_parallel import parallel_forward

    def route(logits, top_k, normalize):
        selected, ids = logits.topk(top_k, dim=-1)
        weights = (
            selected.softmax(-1) if normalize else logits.softmax(-1).gather(1, ids)
        )
        return weights, ids, torch.bincount(ids.flatten(), minlength=logits.shape[1])

    monkeypatch.setattr(router, "route_op", route)
    generator = torch.Generator().manual_seed(88)
    hidden = torch.randn(2, 5, 16, generator=generator).bfloat16()
    weight = torch.randn(7, 16, generator=generator).bfloat16()

    class Experts:
        config = SimpleNamespace(
            model_dim=16,
            top_k=2,
            renormalize_topk=True,
            router_weight_grad_dtype=torch.float32,
        )
        router_weight = weight

        def __call__(self, value, *, expert_ids, routing_weights):
            self.ids, self.weights = expert_ids, routing_weights
            return value

    experts = Experts()
    _, counts, sums = parallel_forward(experts, hidden)
    expected_logits = hidden.float().flatten(0, 1) @ weight.float().t()
    selected, ids = expected_logits.topk(2, dim=-1)
    torch.testing.assert_close(experts.weights, selected.softmax(-1))
    torch.testing.assert_close(sums, expected_logits.softmax(-1).sum(0))
    torch.testing.assert_close(counts, torch.bincount(ids.flatten(), minlength=7))


@pytest.mark.parametrize("borrowed", [False, True])
def test_constructor_failure_releases_only_owned_resources(
    architecture, fake_ep, borrowed
):
    cls, config = architecture
    fake_ep.fail_at = 1
    options = {"buffer": fake_ep.Buffer()} if borrowed else {"token_capacity": 256}
    previous_dtype = torch.get_default_dtype()
    with pytest.raises(RuntimeError, match="injected construction failure"):
        cls(
            config,
            ep_group=object(),
            device="cuda:0",
            parameter_device="cpu",
            **options,
        )
    assert torch.get_default_dtype() == previous_dtype
    assert fake_ep.layers[0].closed
    assert sum(kind == "destroy" for kind, _ in fake_ep.events) == (
        0 if borrowed else 1
    )


@pytest.mark.parametrize(
    "options,match",
    [
        ({"token_capacity": 256}, "requires an EP process group"),
        ({"ep_group": object()}, "exactly one"),
        ({"ep_group": object(), "token_capacity": 0}, "positive integer"),
        (
            {"ep_group": object(), "token_capacity": 256, "buffer": object()},
            "exactly one",
        ),
        ({"ep_group": object(), "token_capacity": 256, "device": "cpu"}, "CUDA device"),
        ({"ep_group": object(), "token_capacity": 256, "dtype": torch.float32}, "BF16"),
    ],
)
def test_invalid_ep_setup_fails_before_allocating(
    architecture, fake_ep, options, match
):
    cls, config = architecture
    options = {"device": "cuda:0", "parameter_device": "cpu", **options}
    with pytest.raises(ValueError, match=match):
        cls(config, **options)
    assert not fake_ep.buffers and not fake_ep.layers


def test_empty_ep_model_fails_before_allocating(fake_ep):
    config = OLMoEConfig(0, 128, 4, 2, 32, 4, 2, 128, 97, max_seq_len=16)
    with pytest.raises(ValueError, match="at least one layer"):
        OLMoE(
            config,
            ep_group=object(),
            token_capacity=256,
            device="cuda:0",
            parameter_device="cpu",
        )
    assert not fake_ep.buffers and not fake_ep.layers


def test_qwen_te_backend_selection(fake_ep, monkeypatch):
    import mlops.expert_parallel as ep

    calls = []
    original = ep.__dict__["QuackMoE"]

    def te_layer(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setitem(ep.__dict__, "TEMoE", te_layer)
    config = replace(
        Qwen3MoEConfig(),
        n_layers=2,
        d_model=128,
        n_heads=4,
        n_kv_heads=2,
        head_dim=32,
        n_experts=4,
        top_k=2,
        d_ff_expert=128,
        vocab_size=97,
        max_seq_len=16,
    )
    model = Qwen3MoE(
        config,
        ep_group=object(),
        token_capacity=256,
        device="cuda:0",
        parameter_device="cpu",
        dtype=torch.bfloat16,
        expert_backend="te",
    )
    assert len(calls) == 2
    assert all(layer.config.share_expert_banks for layer in fake_ep.layers)
    assert all(
        type(layer.config).__module__.endswith("transformer_engine.config")
        for layer in fake_ep.layers
    )
    model.close()
