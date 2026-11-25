"""Inkling loop state, runtime packing and checkpoint geometry regressions."""

from dataclasses import replace

import pytest
import torch
from mlops.dispatch import weight_gradients_at
from torch.fx.experimental.proxy_tensor import make_fx

from shadowspill_workloads.lora import configure_lora
from shadowspill_workloads.models.inkling2_mini.config import PRESETS
from shadowspill_workloads.models.inkling2_mini.implementations.mlops import Model
from shadowspill_workloads.models.inkling2_mini.implementations.mlops.attention import (
    ShortConv,
)
from shadowspill_workloads.objectives.head.execution import loss as token_objective


@pytest.mark.parametrize("fp32_projection", [True, False])
def test_softcapped_training_head_consumes_each_chunk_and_preserves_reductions(
    fp32_projection,
):
    torch.manual_seed(120)
    model = Model(replace(PRESETS["tiny"], head_fp32_projection=fp32_projection))
    model.lm_head.weight.requires_grad_(False)
    tokens = torch.tensor([[5, 7, 4, 3, 8, 1, 0]])
    targets = torch.tensor([[7, 4, -100, 8, 1, 0, 2]])
    hidden = model.hidden(tokens)
    logits = model.lm_head(hidden).flatten(0, -2)
    per_token = torch.nn.functional.cross_entropy(
        logits, targets.flatten(), reduction="none"
    )
    mask = targets != -100
    weights = torch.linspace(0.04, 0.22, 7).reshape_as(targets)
    features = dict(head=model.lm_head, head_hidden=hidden, head_targets=targets)
    got, values = token_objective(
        features, dict(loss_mask=mask, weights=weights), {}, policy=False
    )
    expected = (per_token * weights.flatten()).sum()
    torch.testing.assert_close(got, expected)
    torch.testing.assert_close(values.flatten(), -per_token)
    assert not values.requires_grad
    saved = [v for v in got.grad_fn.saved_tensors if v is not None]
    assert len(saved) == 1 and saved[0].shape == hidden.shape
    for reduction in ("sum", "mean", "none"):
        torch.testing.assert_close(
            model.loss(tokens, targets, reduction=reduction, head_chunk_size=3),
            torch.nn.functional.cross_entropy(
                logits, targets.flatten(), reduction=reduction
            ),
        )
    ignored = model.loss(tokens, torch.full_like(targets, -100))
    assert ignored.isnan()
    assert torch.autograd.grad(ignored, model.norm.weight)[0].count_nonzero() == 0


@pytest.mark.parametrize("mode", ["full", "lora", "lora_and_base"])
@pytest.mark.parametrize("fp32_projection", [True, False])
def test_selected_head_preserves_softcap_and_chunked_parameter_gradients(
    mode, fp32_projection
):
    torch.manual_seed(543)
    config = replace(
        PRESETS["tiny"], head_fp32_projection=fp32_projection, logit_cap=0.75
    )
    model = Model(config)
    weight = model.lm_head.weight
    names = [] if mode == "full" else ["lm_head"]
    configure_lora(
        model,
        targets=names,
        rank=3,
        alpha=5,
        head="full" if mode != "lora" else "frozen",
        factor_dtype=torch.float32,
    )
    head = model.lm_head
    assert head.weight is weight
    wrt = {k: v for k, v in head.named_parameters() if v.requires_grad}
    expected_names = ({"weight"} if mode != "lora" else set()) | (
        {"lora_a", "lora_b"} if mode != "full" else set()
    )
    assert set(wrt) == expected_names
    x = torch.randn(1, 9, config.d_model, requires_grad=True)
    labels = torch.arange(9)[None, :] + 1
    labels[0, 3] = -100
    mask = labels != -100
    row_weights = torch.linspace(0.03, 0.19, 9)[None, :]
    raw = torch.nn.functional.linear(x, weight)
    if mode != "full":
        with torch.no_grad():
            head.lora_b.normal_(std=0.3)
        raw = raw + ((x @ head.lora_a.T) @ head.lora_b.T) * (5 / 3)
    logits = config.logit_cap * torch.tanh(raw / config.logit_cap)
    torch.testing.assert_close(head(x), logits)
    expected_values = -torch.nn.functional.cross_entropy(
        logits[0], labels[0], reduction="none"
    ).reshape_as(labels)
    with weight_gradients_at(torch.float32):
        actual, values = head.objective(
            x,
            labels,
            objective="cross_entropy",
            inputs=(mask, row_weights),
            chunk_size=4,
        )
    expected = -(row_weights * expected_values).sum()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(values, expected_values)
    torch.testing.assert_close(
        head.loss(x, labels, chunk_size=4, reduction="sum"), -expected_values.sum()
    )
    torch.testing.assert_close(
        head.loss(x, labels, chunk_size=4, valid_rows=8), -expected_values.sum() / 8
    )
    seeds = actual.grad_fn.saved_tensors
    assert [s is not None for s in seeds] == [
        True,
        "weight" in wrt,
        "lora_a" in wrt,
        "lora_b" in wrt,
    ]
    parameters = (x, *wrt.values())
    for found, wanted in zip(
        torch.autograd.grad(actual, parameters),
        torch.autograd.grad(expected, parameters),
        strict=True,
    ):
        torch.testing.assert_close(found, wanted)
        assert found.count_nonzero() > 0

    head_only, _ = head.objective(
        x.detach(),
        labels,
        objective="cross_entropy",
        inputs=(mask, row_weights),
        chunk_size=4,
    )
    assert head_only.grad_fn.saved_tensors[0] is None


@pytest.mark.parametrize("dynamic", [False, True])
def test_convolution_cotangents_use_disjoint_tap_slices(dynamic):
    torch.manual_seed(172)
    width, taps = 12, 3
    options = {"dynamic_input": width, "heads": 3} if dynamic else {}
    conv = ShortConv(width, taps, **options)
    x, source, grad = torch.randn(3, 1, 8, width)
    positions = torch.tensor([[0, 1, 2, 0, 1, 2, 3, 4]])

    def objective(value):
        return (conv(value, positions, source) * grad).sum()

    derivative = torch.func.grad(objective)
    graph = make_fx(derivative)(x)
    # An overlapping unfold lowers to an atomic index add in the CUDA
    # backward. Its last-bit changes can change a subsequent MXFP8 cotangent.
    targets = {n.target for n in graph.graph.nodes}
    assert torch.ops.aten.unfold_backward.default not in targets
    assert torch.ops.aten._unsafe_index_put.default not in targets
    # Check that gradients at a packed boundary also agree with independent
    # sequences, including dynamically weighted taps.
    parts = []
    for start, stop in [(0, 3), (3, 8)]:
        part = x[:, start:stop].detach().requires_grad_()
        output = conv(part, positions[:, start:stop], source[:, start:stop])
        (dx,) = torch.autograd.grad(output, part, grad[:, start:stop])
        parts.append(dx)
    torch.testing.assert_close(derivative(x), torch.cat(parts, dim=1))


@pytest.mark.parametrize("query_block", [0, 3])
def test_repeated_core_accumulates_each_occurrence_and_packed_sequences_reset(
    query_block,
):
    torch.manual_seed(185)
    model = Model(replace(PRESETS["tiny"], attention_query_chunk_size=query_block))
    tokens = torch.tensor([[8, 15, 17, 0, 2, 1, 4, 7]])
    lengths = torch.tensor([3, 0, 5], dtype=torch.int32)
    model.layers[1].attn_query.data.normal_()
    model.middle_cycle_cross_pass_queries.data.normal_()
    core = model.layers[1].attn.wq_du
    occurrences = []

    def save(module, inputs, output):
        output.retain_grad()
        occurrences.append((inputs[0], output))

    with core.register_forward_hook(save):
        packed = model(tokens, lengths)
    assert len(occurrences) == model.config.passes == 3
    torch.testing.assert_close(
        packed, torch.cat([model(tokens[:, :3]), model(tokens[:, 3:])], dim=1)
    )
    labels = tokens.roll(-1, dims=1)
    loss = torch.nn.functional.cross_entropy(packed[0], labels[0])
    loss.backward()
    contributions = [
        out.grad.flatten(0, 1).T @ x.detach().flatten(0, 1) for x, out in occurrences
    ]
    assert all(g.abs().max() > 0 for g in contributions)
    torch.testing.assert_close(core.weight.grad, sum(contributions))
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_later_passes_receive_messages_from_previous_pass():
    torch.manual_seed(100)
    config = PRESETS["tiny"]
    model = Model(config)
    tokens = torch.tensor([[1, 3, 5, 7]])
    # Zero gates make every pass reproduce the first one.
    for layer in model.layers[config.core_start : config.core_end]:
        for gate in layer.controllers.values():
            gate.gain_weight.data.zero_()
            gate.forget_weight.data.zero_()
    single = Model(replace(config, passes=1))
    single.load_state_dict(model.state_dict())
    torch.testing.assert_close(model(tokens), single(tokens))
    with torch.no_grad():
        model.layers[config.core_start].controllers["attn"].gain_weight.normal_(std=0.5)
    assert not torch.allclose(model(tokens), single(tokens))


def test_published_core_has_one_parameter_set_per_physical_layer():
    with torch.device("meta"):
        model = Model(PRESETS["published"])
    assert len(model.layers) == 24
    assert (
        model.config.n_layers
        + (model.config.passes - 1) * (model.config.core_end - model.config.core_start)
        == 48
    )
    assert model.lm_head.out_features == 200058 and model.embed.num_embeddings == 201024
    assert model.lm_head.weight is not model.embed.weight


def test_published_attention_only_lora_leaves_all_experts_frozen():
    from shadowspill_workloads.lora import configure_lora

    targets = [f"layers.*.attn.{name}" for name in ("wq_du", "wk_dv", "wv_dv", "wo_ud")]
    with torch.device("meta"):
        model = configure_lora(
            Model(PRESETS["published"]), rank=2, alpha=2, targets=targets
        )
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    assert len(trainable) == 192
    assert sum(p.numel() for p in trainable.values()) == 835584
    assert all(".attn." in name and ".lora_" in name for name in trainable)
    assert all(not hasattr(layer.mlp, "lora_config") for layer in model.layers)


@pytest.mark.parametrize(
    "changes",
    [
        {"core_start": 5},
        {"passes": 0},
        {"n_kv_heads": 7},
        {"attention_query_chunk_size": -1},
        {"attention_query_chunk_size": True},
        {"attention_backend": "unknown"},
        {"attention_backend": "fused", "attention_query_chunk_size": 64},
    ],
)
def test_invalid_loop_and_attention_geometry_is_rejected(changes):
    with pytest.raises(ValueError):
        replace(PRESETS["published"], **changes)


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
def test_checkpoint_decodes_unaligned_ranges_across_experts_without_scale_rounding(
    tmp_path, fmt
):
    from mlops.sequential_moe.precision.safetensors import SafeTensors
    from safetensors.torch import save_file

    from shadowspill_workloads.models.inkling2_mini.checkpoint import _decoded

    e, rows, cols = 2, 3, 64
    if fmt == "nvfp4":
        codes = (torch.arange(e * rows * cols).reshape(e, rows, cols) % 16).byte()
        data = codes[..., ::2] | (codes[..., 1::2] << 4)
        scales = (
            torch.linspace(0.1, 2.0, e * rows * 4)
            .reshape(e, rows, 4)
            .to(torch.float8_e4m3fn)
        )
        amax = torch.tensor([1.0137, 4.0921])
        denominator = torch.tensor([13.719])
        value = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])[
            (codes & 7).long()
        ]
        expected = (
            value
            * torch.where(codes < 8, 1, -1)
            * scales.float().repeat_interleave(16, -1)
            * (amax / denominator)[:, None, None]
        )
        extra = {"bank_amax": amax, "bank_outer_scale_denominator": denominator}
    else:
        data = (
            torch.arange(e * rows * cols)
            .reshape(e, rows, cols)
            .remainder(19)
            .to(torch.float8_e4m3fn)
        )
        scales = torch.arange(122, 134, dtype=torch.uint8).reshape(e, rows, 2)
        expected = torch.ldexp(
            data.float(), (scales.int() - 127).repeat_interleave(32, -1)
        )
        extra = {}
    save_file(
        {"bank": data, "bank_scale": scales, **extra}, tmp_path / "model.safetensors"
    )
    with SafeTensors(tmp_path) as reader:
        source = _decoded("bank", fmt)
        dst = torch.empty_like(expected).view(-1)
        for offset in range(0, dst.numel(), 37):
            source.copy_into(reader, offset, dst[offset : offset + 37])
        torch.testing.assert_close(dst.view_as(expected), expected, rtol=0, atol=0)
        torch.testing.assert_close(source.read(reader), expected, rtol=0, atol=0)
        bf16 = torch.empty(dst.shape, dtype=torch.bfloat16)
        for offset in range(0, bf16.numel(), 37):
            source.copy_into(reader, offset, bf16[offset : offset + 37])
        assert torch.equal(bf16.view_as(expected), expected.bfloat16())


@pytest.mark.parametrize("storage", ["packed", "bf16"])
@pytest.mark.parametrize("layer", [0, 3])
def test_bf16_storage_selects_grouped_projection_and_preserves_lora(storage, layer):
    from shadowspill_workloads.lora import configure_lora

    c = replace(
        PRESETS["tiny"],
        expert_formats=("nvfp4", "nvfp4", "mxfp8", "mxfp8"),
        expert_storage=storage,
    )
    # The outer model's FP32 dtype also exercises explicit BF16 bank storage.
    model = configure_lora(Model(c), rank=2, alpha=2)
    offsets = torch.tensor([0, 2, 2, 3, 4], dtype=torch.int32)
    x = torch.randn(4, c.d_model, requires_grad=True)
    expert = model.layers[layer].mlp
    if storage == "packed":
        expert.w13.zero_()
        expert.w13_scale.zero_()
        if hasattr(expert, "w13_amax"):
            expert.w13_amax.fill_(1)
            expert.w13_outer_scale_denominator.fill_(1)
    else:
        assert expert.w13.dtype == expert.w2.dtype == torch.bfloat16
        assert not hasattr(expert, "w13_scale")
    assert not expert.w13.requires_grad and expert.lora_gate_up_b.requires_grad

    def project(values):
        return expert.projection(values, "w13", "gate_up", offsets)

    graph = make_fx(project, tracing_mode="fake", _allow_non_fake_inputs=True)(x)
    ops = {n.target for n in graph.graph.nodes}
    assert (torch.ops.mlops.groupwise_expert_linear.default in ops) == (
        storage == "packed"
    )
    assert torch.ops.mlops.lora_grouped_linear_fwd.default in ops
    assert torch.ops.mlops.microscaled_expert_linear.default not in ops
    dx, db = torch.autograd.grad(project(x).sum(), (x, expert.lora_gate_up_b))
    assert dx.isfinite().all() and db.isfinite().all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
def test_auto_and_forced_bf16_dispatch_on_capture_device(fmt):
    from shadowspill_workloads.models.inkling2_mini.implementations.mlops import experts

    c = replace(PRESETS["tiny"], expert_storage="packed", expert_formats=(fmt,) * 4)
    x = torch.zeros(4, c.d_model, device="cuda", dtype=torch.bfloat16)
    offsets = torch.arange(5, device="cuda", dtype=torch.int32)

    def projection_ops(compute):
        layer = experts.Experts(replace(c, expert_compute=compute), 0).cuda()

        def project(values):
            return layer.projection(values, "w13", "gate_up", offsets)

        trace = make_fx(project, tracing_mode="fake", _allow_non_fake_inputs=True)(x)
        return {node.target for node in trace.graph.nodes}

    bf16 = torch.ops.mlops.groupwise_expert_linear.default
    supported = torch.cuda.get_device_capability() >= (
        (9, 0) if fmt == "nvfp4" else (8, 9)
    )
    assert bf16 in projection_ops("bf16")
    if supported:
        assert projection_ops("auto") == projection_ops("checkpoint")
        assert bf16 not in projection_ops("auto")
    else:
        assert projection_ops("auto") == projection_ops("bf16")
        with pytest.raises(RuntimeError, match="checkpoint"):
            projection_ops("checkpoint")


def test_checkpoint_rejects_a_different_activation_rule():
    from shadowspill_workloads.models.inkling2_mini.config import validate_recipe

    key = "model.llm.layers.0.mlp.experts.w13_weight"
    recipe = {
        "quant_method": "inkling2",
        "model": {
            "parameter_groups": [
                {
                    "tensors": [key],
                    "compute_dtype": "nvfp4",
                    "weight_transform": "nvfp4_nearest",
                    "required_activation_quantization": {
                        "scale_selection_rule": "amax"
                    },
                }
            ]
        },
    }
    with pytest.raises(ValueError, match="expert recipe disagrees"):
        validate_recipe(recipe, ("nvfp4",), {key: {"dtype": "U8"}})
