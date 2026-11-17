"""Fixed geometry of GLM's installed sparse, recurrent and FP8 LoRA operators.

Packed metadata lives on the device. KDA receives fixed-capacity cumulative and
chunk arrays, with 64-row intra-chunk matrices and histories sized by allocated
chunk rows. Empty sequence slots change work, never storage. The pooled indexer
allocates T*(top_k+pool-1) indices, floor(T/pool) pooled rows and a shape-bounded
score workspace. It consumes boundary values on-device; the host-boundary cuDNN
indexer is intentionally excluded from this contract.

Sparse attention pads queries/KV by fixed 64-column tails and an optional sentinel
row. Index padding follows a fixed provider tile. Its LSE, delta and FP32 KV
gradients are geometry-sized. Hopper's FlashMLA selection/LSE convention is saved
by a separate preparation factory, so execution never performs its GPU probe.

Four-stream mHC uses T*4/T*16 coefficients and input-sized outputs/gradients.
DeepGEMM preprocessing allocates fixed split*T*24 and split*T partials; backward
allocates fixed coefficient partials and input gradients. Frozen projections
decode exactly their declared WeightSpec shape. Native block-FP8 expert GEMMs
allocate one-byte row operands plus row scales and fixed GEMM outputs; factor
padding and products depend on LoRA rank, not expert counts in the offset values.

EP LoRA uses the existing MoonEP P=T*K+2*q*127 dispatch capacity. Its frozen base
and factor publication banks belong to the admitted Resource; execution allocates
fixed P-row activations, rank-padded LoRA intermediates and local factor gradients.
Native block storage needs only one FP8 orientation. No FP8-current conversion,
host count readback, fused shared-expert EP or multi-chunk EP path is covered here.

Declarations identify allocation invariance, not byte bounds or numerical
qualification. The ordinary compiler/profiler and resource admission still apply.
All implementations and external providers are pinned by the model definition.
"""

from shadowspill.preparation import OperatorPreparation
from shadowspill_workloads.definitions.moe_provider import (
    allocations as moe_allocations,
)

SPARSE_FACTORY = "mlops.providers.builtin.sparse_latent_preparation:factory"
CONTRACT = "shadowspill/glm-runtime-packed-allocation@1"


def contracts(config):
    result = {
        **moe_allocations(config.compute_dtype),
        **{
            f"torch.ops.mlops.{name}_{direction}.default": CONTRACT
            for name in (
                "kimi_delta_attention_fla",
                "kimi_delta_attention_fla_ieee",
                "packed_swiglu_builtin_triton",
                "sparse_latent_attention_tilelang",
                "sparse_latent_attention_hopper",
                "sparse_latent_rope_tilelang",
                "sparse_latent_rope_hopper",
            )
            for direction in ("fwd", "bwd")
        },
        **{
            f"torch.ops.mlops.{name}.default": CONTRACT
            for name in (
                "pooled_topk_builtin_triton",
                "causal_conv_silu_qkv_fla_fwd",
                "causal_conv_silu_qkv_input_bwd",
                "mhc_mix_four",
                "mhc_post_four",
                "mhc_post_four_input_grads",
                "mhc_post_four_coefficient_grads",
                "mhc_pre_deepgemm",
                "mhc_pre_deepgemm_backward",
            )
        },
        "torch.ops.mlops_seqmoe.decode_weight.default": CONTRACT,
        SPARSE_FACTORY: "shadowspill/sparse-forward-prepared-activation@1",
    }
    if config.architecture["expert_storage"] == "fp8_block":
        result.update(
            {
                f"torch.ops.mlops.lora_sonicmoe_{direction}_{phase}.default": (
                    "shadowspill/sonic-lora-block-fp8-allocation@1"
                )
                for direction in ("up", "down")
                for phase in ("fwd", "bwd")
            }
        )
    if config.implementation == "mlops_ep":
        result.update(
            {
                f"torch.ops.mlops_ep_quack_lora.{name}.default": (
                    "shadowspill/quack-ep-lora-allocation@1"
                )
                for name in (
                    "forward",
                    "backward",
                    "physical_forward",
                    "physical_backward",
                )
            }
        )
    return result


def preparations(config):
    from shadowspill_workloads.definitions.moe_provider import preparations as sonic

    return (
        OperatorPreparation(factory=SPARSE_FACTORY),
        *(
            sonic(config.compute_dtype)
            if config.implementation == "mlops"
            and config.architecture["expert_storage"] == "bf16"
            else ()
        ),
    )
