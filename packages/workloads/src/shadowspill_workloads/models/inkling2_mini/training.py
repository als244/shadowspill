"""Inkling setup reuses the packed-text component and standard execution hooks."""

from contextlib import contextmanager
from dataclasses import asdict, replace

from shadowspill_workloads.components import training_options
from shadowspill_workloads.components.decoder import configuration
from shadowspill_workloads.components.decoder import register as register_decoder
from shadowspill_workloads.components.moe_allocations import CONTRACTS

from . import config, partition


class Settings(configuration.Settings):
    family = "inkling2_mini"
    catalog = config


def build(settings, context):
    settings = training_options.for_build(settings, context)
    geometry = dict(settings.architecture)
    if geometry["expert_storage"] == "auto":
        use_packed = settings.lora and set(geometry["expert_formats"]) != {"dense"}
        geometry["expert_storage"] = "packed" if use_packed else "dense"
    if not settings.lora and geometry["expert_storage"] == "packed":
        raise ValueError(
            "packed Inkling experts are frozen; use dense storage for full training"
        )
    if geometry["expert_compute"] == "auto" and geometry["expert_storage"] != "packed":
        geometry["expert_compute"] = "bf16"
    if (
        geometry["expert_compute"] == "checkpoint"
        and geometry["expert_storage"] != "packed"
    ):
        raise ValueError(
            "checkpoint expert arithmetic requires the frozen, packed banks"
        )
    settings = replace(settings, architecture=geometry)
    from shadowspill.training.components import ModelBuild

    # Counts select the rows processed by each expert. The packed kernel has one output
    # per assigned row and a tile prefix sized by the number of experts.
    return ModelBuild(
        asdict(settings),
        grad_dtype=settings.grad_dtype,
        allocation_invariants={
            **CONTRACTS,
            "torch.ops.mlops.attention_residual_backward.default": (
                "shadowspill/attention-residual-backward-fixed-shapes@1"
            ),
            "torch.ops.mlops.sigmoid_gated_value.default": (
                "shadowspill/fixed-sigmoid-gated-value@1"
            ),
            "torch.ops.mlops.residual_tap_mix.default": "shadowspill/fixed-tap-mix@1",
            "torch.ops.mlops.packed_residual_conv.default": (
                "shadowspill/fixed-packed-residual-conv@1"
            ),
            "torch.ops.mlops.packed_residual_conv_dx.default": (
                "shadowspill/fixed-packed-residual-conv-dx@1"
            ),
            "torch.ops.mlops.routed_softcapped_swiglu_backward.default": (
                "shadowspill/fixed-routed-activation-backward@1"
            ),
            "torch.ops.mlops.routed_softcapped_swiglu.default": (
                "shadowspill/fixed-routed-activation@1"
            ),
            "torch.ops.mlops.projected_relative_gqa.default": (
                "shadowspill/fixed-group-projected-gqa@1"
            ),
            "torch.ops.mlops.projected_relative_gqa_backward.default": (
                "shadowspill/fixed-group-projected-gqa-backward@1"
            ),
            "torch.ops.mlops.relative_gqa_fused.default": (
                "shadowspill/fixed-relative-fused-attention@1"
            ),
            "torch.ops.mlops.relative_gqa_fused_backward.default": (
                "shadowspill/fixed-relative-fused-attention-backward@2"
            ),
            "torch.ops.mlops.relative_gqa_query_blocks.default": (
                "shadowspill/fixed-query-block-attention@1"
            ),
            "torch.ops.mlops.relative_gqa_query_blocks_backward.default": (
                "shadowspill/fixed-query-block-attention-backward@2"
            ),
            "torch.ops.mlops.l2_clip_scale.default": (
                "shadowspill/fixed-l2-clip-scale@1"
            ),
            "torch.ops.mlops.stable_normalized_sigmoid_topk.default": (
                "shadowspill/fixed-router-sort@1"
            ),
            "torch.ops.mlops.rms_gain_cast.default": (
                "shadowspill/fixed-rms-gain-cast@1"
            ),
            "torch.ops.mlops.rms_normalize_fp32.default": (
                "shadowspill/fixed-rms-shape@1"
            ),
            "torch.ops.mlops.attention_residual.default": (
                "shadowspill/attention-residual-fixed-shapes@1"
            ),
            # ATen topk allocates two [tokens, k] outputs; CUDA selection scratch is
            # sized by input geometry, dimension and k, independent of the selected IDs.
            "torch.ops.aten.topk.default": "shadowspill/inkling-fixed-topk@1",
            "torch.ops.mlops.groupwise_expert_linear.default": (
                "shadowspill/groupwise-expert-allocation@2"
            ),
            "torch.ops.mlops.microscaled_expert_linear.default": (
                "shadowspill/microscaled-expert-allocation@1"
            ),
            "torch.ops.mlops.routed_microscaled_expert_linear.default": (
                "shadowspill/routed-microscaled-expert-allocation@1"
            ),
            "torch.ops.mlops.routed_microscaled_lora_up.default": (
                "shadowspill/routed-microscaled-lora-allocation@1"
            ),
            "torch.ops.mlops.block_scaled_mxfp8_expert_linear.default": (
                "shadowspill/microscaled-expert-allocation@1"
            ),
            "torch.ops.mlops.microscaled_expert_dgrad.default": (
                "shadowspill/microscaled-expert-allocation@1"
            ),
            "torch.ops.aten.sort.stable": "shadowspill/fixed-router-sort@1",
        },
    )


@contextmanager
def capture(settings):
    from torch._inductor import config as inductor

    with (
        training_options.capture_context(settings),
        inductor.patch(
            {"emulate_precision_casts": True, "eager_numerics.division_rounding": True}
        ),
    ):
        yield


def register(registry):
    partition.register(registry)
    register_decoder(
        registry,
        Settings,
        build=build,
        capture_context=capture,
        sources=("components/moe_allocations.py", "components/hybrid_allocations.py"),
    )
