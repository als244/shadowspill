"""The settings of one Megatron-Core case, shared by the launcher and the
trainer.

The engine trains NVIDIA's prebuilt model of the catalog model's
architecture with Megatron Bridge's training loop over Megatron-Core. The
model is the one Bridge builds from the case's Hugging Face configuration,
the catalog model's from its family's ``to_huggingface`` or a named
checkpoint's own, whose weights Bridge then imports; or the model of the
Bridge recipe the case names, which must have the same architecture. The
shared settings (the model, the geometry, the precision contract, the run)
are applied on top of Bridge's defaults or the recipe's values; Megatron's
own options (``OPTIONS``) are settings under Megatron's names, each left to
Bridge or the recipe unless the case sets it. ``README.md`` says what each
does and which go together.
"""

from __future__ import annotations

import dataclasses
import json
from importlib import import_module
from pathlib import Path
from typing import Any

from baselines.common import settings as common
from baselines.common.geometry import Geometry
from baselines.common.precision import Capabilities, Unsupported
from baselines.common.settings import GEOMETRY_FIELDS, MODEL_FIELDS, PRECISION_FIELDS

#: The sources whose digests identify this engine in a manifest and a row.
SOURCE_PATHS = (
    "baselines/__init__.py",
    "baselines/common",
    "baselines/engines/megatron_core",
)

#: The environment beside the launcher's that the trainer runs in.
ENVIRONMENT = "nvidia-training"

#: The model block's name for NVIDIA's implementation.
IMPLEMENTATION = "megatron_core"

#: What Megatron-Core honors of the precision contract. Under DDP its
#: optimizers keep master copies: float32 ones, or, through its
#: precision-aware optimizer and Transformer Engine's fused Adam (which
#: rounds to nearest), float32 or float16 masters with moments at float32,
#: float16 or bfloat16. Megatron-FSDP can step the compute weights
#: themselves (``megatron_fsdp_main_params_dtype`` none), the one way to
#: keep no master, with Transformer Engine's fused Adam keeping the moments at
#: float32. Gradients accumulate at the model dtype or float32; with
#: ``gradient_accumulation_fusion`` a weight-gradient matmul adds into a
#: bfloat16 accumulation as it writes it, the contract's fused narrow-gradient
#: accumulation. A float16 model would need loss scaling the contract does
#: not name.
CAPABILITIES = Capabilities(
    IMPLEMENTATION,
    model_dtypes=("bfloat16", "float32"),
    master_dtypes=(None, "float32", "float16"),
    grad_dtypes=(None, "float32", "bfloat16"),
    opt_state_dtypes=(None, "parameter", "float32", "float16", "bfloat16"),
    parameter_roundings=(None,),
    opt_state_roundings=(None,),
    fuse_narrow_gradient_accumulation=True,
)


@dataclasses.dataclass(frozen=True)
class Option:
    """One of Megatron's options, as a case sets it: its name in Megatron,
    the Bridge configuration that holds it (``model``, the provider and its
    ``TransformerConfig``; ``optimizer``; ``ddp``; ``dist``; ``mixed``, the
    mixed-precision configuration Bridge copies onto the others; ``comm``,
    its communication-overlap configuration; ``environment``, a variable the
    launcher sets in the trainer's environment), joined by ``+`` where Bridge
    keeps two in step, the type of its value, what it does, and its allowed
    values. ``form`` says how a string is read: ``names``, a comma-separated
    list of the allowed values; ``integers``, a comma-separated list of
    integers; ``dtype``, a dtype's name."""

    name: str
    section: str
    kind: type
    help: str
    choices: tuple = ()
    form: str = "value"


RECOMPUTE_MODULES = (
    "core_attn",
    "moe_act",
    "layernorm",
    "mla_up_proj",
    "mlp",
    "moe",
    "shared_experts",
    "gdn_norm_out",
    # Of model families no catalog model is: Gated DeltaProduct layers'
    # projections, which recompute nothing in another model, and hybrid
    # models' hyper-connections, shortcut blocks and residual streams, which
    # Megatron refuses without them.
    "gdp_in_proj",
    "gdp_qkv",
    "mhc",
    "shortcut_pre_mlp_layernorm",
    "residual_stream",
)
OFFLOAD_MODULES = (
    "attn_norm",
    "qkv_linear",
    "core_attn",
    "attn_proj",
    "mlp_norm",
    "expert_fc1",
    "moe_act",
    "fused_group_mlp",
    # Gated DeltaProduct layers' and hybrid shortcut blocks'; Megatron
    # refuses the shortcut blocks' without them.
    "gdp_qkv",
    "shortcut_post_norm",
)
#: What selective recomputation recomputes when a case names no modules:
#: Megatron's own default, core attention, with the norms, and in a mixture
#: of experts the experts' activation too.
DEFAULT_RECOMPUTE_MODULES = ("core_attn", "layernorm")
MIXTURE_RECOMPUTE_MODULES = ("moe_act",)
SHARDING = ("no_shard", "optim", "optim_grads", "optim_grads_params")
DTYPES = ("bfloat16", "float16", "float32")

#: Every Megatron option a case can set, by subject; ``README.md`` describes
#: each and what goes together.
OPTIONS = (
    # Parallelism.
    Option(
        "tensor_model_parallel_size",
        "model",
        int,
        "ranks each layer's weights and activations are split across",
    ),
    Option(
        "pipeline_model_parallel_size",
        "model",
        int,
        "stages the layers are divided into, one group of ranks each",
    ),
    Option(
        "virtual_pipeline_model_parallel_size",
        "model",
        int,
        "model chunks per pipeline rank, interleaved to shrink the pipeline"
        " bubble; needs pipeline parallelism",
    ),
    Option(
        "num_layers_in_first_pipeline_stage",
        "model",
        int,
        "layers on the first pipeline stage, the rest divided evenly",
    ),
    Option(
        "num_layers_in_last_pipeline_stage",
        "model",
        int,
        "layers on the last pipeline stage, the rest divided evenly",
    ),
    Option(
        "account_for_embedding_in_pipeline_split",
        "model",
        bool,
        "count the embedding as a layer when dividing the stages",
    ),
    Option(
        "account_for_loss_in_pipeline_split",
        "model",
        bool,
        "count the loss as a layer when dividing the stages",
    ),
    Option(
        "context_parallel_size",
        "model",
        int,
        "ranks each sequence is split across, attention exchanging keys and"
        " values among them",
    ),
    Option(
        "cp_comm_type",
        "model",
        str,
        "context parallelism's exchange: p2p (a ring, overlapped with"
        " attention), all_gather, a2a (Ulysses-style) or a2a+p2p"
        " (hierarchical)",
        ("p2p", "all_gather", "a2a", "a2a+p2p"),
    ),
    Option(
        "hierarchical_context_parallel_sizes",
        "model",
        str,
        "with a2a+p2p, the a2a degree and the p2p degree, whose product is"
        " context_parallel_size",
        form="integers",
    ),
    Option(
        "expert_model_parallel_size",
        "model",
        int,
        "ranks the routed experts are distributed over, out of the data-parallel ranks",
    ),
    Option(
        "expert_tensor_parallel_size",
        "model",
        int,
        "ranks each expert's weights are split across (Bridge's default is"
        " 1 under expert parallelism)",
    ),
    Option(
        "sequence_parallel",
        "model",
        bool,
        "split the activations outside the tensor-parallel matmuls along"
        " the sequence; needs tensor parallelism",
    ),
    Option(
        "overlap_p2p_comm",
        "model",
        bool,
        "overlap pipeline sends and receives with compute; not with batch_p2p_comm",
    ),
    Option(
        "batch_p2p_comm",
        "model",
        bool,
        "batch the pipeline's sends and receives in one call",
    ),
    Option(
        "deallocate_pipeline_outputs",
        "model",
        bool,
        "free a stage's output once it is sent to the next stage",
    ),
    Option(
        "microbatch_group_size_per_vp_stage",
        "model",
        int,
        "microbatches a virtual stage runs at a time (the pipeline size by default)",
    ),
    Option(
        "defer_embedding_wgrad_compute",
        "model",
        bool,
        "defer the embedding's weight-gradient matmuls into the pipeline"
        " flush; needs pipeline parallelism and gradient_accumulation_fusion",
    ),
    Option(
        "wgrad_deferral_limit",
        "model",
        int,
        "microbatches whose embedding weight gradients are deferred, 0 for all",
    ),
    # Data-parallel sharding and communication overlap.
    Option(
        "use_distributed_optimizer",
        "ddp+optimizer",
        bool,
        "shard the optimizer state and the masters across the data-parallel"
        " ranks (ZeRO-1), gradients reduce-scattered and weights"
        " all-gathered; the precision-aware optimizer needs it",
    ),
    Option(
        "num_distributed_optimizer_instances",
        "ddp",
        int,
        "shard the optimizer state within this many groups of the"
        " data-parallel ranks instead of across all of them",
    ),
    Option(
        "overlap_grad_reduce",
        "ddp",
        bool,
        "reduce each bucket of gradients while backward continues",
    ),
    Option(
        "overlap_param_gather",
        "ddp+optimizer",
        bool,
        "gather the updated weight shards while the next forward runs;"
        " needs the distributed optimizer",
    ),
    Option(
        "overlap_param_gather_with_optimizer_step",
        "optimizer",
        bool,
        "start the first bucket's weight gather during the optimizer step",
    ),
    Option(
        "align_param_gather",
        "ddp",
        bool,
        "launch every pipeline stage's weight gathers together",
    ),
    Option(
        "bucket_size",
        "ddp",
        int,
        "parameters per gradient-reduction and weight-gather bucket",
    ),
    Option(
        "average_in_collective",
        "ddp",
        bool,
        "average gradients in the reduction itself instead of dividing first",
    ),
    Option(
        "reduce_scatter_with_fp32_accumulation",
        "ddp",
        bool,
        "send lower-precision gradients but accumulate the reduction in float32",
    ),
    Option(
        "use_megatron_fsdp",
        "ddp+dist",
        bool,
        "shard with Megatron-Core's own FSDP (Megatron-FSDP) in place of its"
        " DDP; data_parallel_sharding_strategy chooses what is sharded, and"
        " it is the one way to step the compute weights without masters",
    ),
    Option(
        "megatron_fsdp_version",
        "ddp",
        int,
        "Megatron-FSDP's implementation, 1 or 2 (2: bfloat16 only, no tensor"
        " or pipeline parallelism, no optimizer offload or FP8)",
        (1, 2),
    ),
    Option(
        "data_parallel_sharding_strategy",
        "ddp",
        str,
        "what Megatron-FSDP shards across the data-parallel ranks:"
        " no_shard; optim, the optimizer state (ZeRO-1); optim_grads, the"
        " gradients too (ZeRO-2); optim_grads_params, the weights too"
        " (ZeRO-3)",
        SHARDING,
    ),
    Option(
        "expert_data_parallel_sharding_strategy",
        "ddp",
        str,
        "the same for the routed experts' parameters; the dense strategy by default",
        SHARDING,
    ),
    Option(
        "outer_dp_sharding_strategy",
        "ddp",
        str,
        "Megatron-FSDP's hybrid sharding (HSDP): the outer data-parallel"
        " group replicates (no_shard) or shards the optimizer state"
        " (optim)",
        ("no_shard", "optim"),
    ),
    Option(
        "fsdp_double_buffer",
        "ddp",
        bool,
        "keep Megatron-FSDP's communication buffers allocated, double buffered",
    ),
    Option(
        "nccl_ub",
        "ddp",
        bool,
        "register the weight and gradient buffers as NCCL user buffers"
        " (SM-efficient collectives)",
    ),
    Option(
        "megatron_fsdp_enable_fine_grained_param_gather",
        "ddp",
        bool,
        "gather Megatron-FSDP's weights in finer units, overlapping more of"
        " the gather with forward",
    ),
    Option(
        "megatron_fsdp_grad_comm_dtype",
        "ddp",
        str,
        "the dtype Megatron-FSDP sends gradients at; the gradients' own by default",
        DTYPES,
        form="dtype",
    ),
    Option(
        "suggested_communication_unit_size",
        "ddp",
        int,
        "elements Megatron-FSDP communicates at once, which also paces its prefetch",
    ),
    Option(
        "init_model_with_meta_device",
        "model",
        bool,
        "build the model on meta and materialize only each rank's shards"
        " (Megatron-FSDP, not with no_shard alone)",
    ),
    Option(
        "tp_comm_overlap",
        "comm",
        bool,
        "overlap tensor-parallel all-gathers and reduce-scatters with the"
        " linear layers' matmuls (Transformer Engine userbuffers); needs"
        " tensor and sequence parallelism",
    ),
    Option(
        "tp_comm_bulk_wgrad",
        "model",
        bool,
        "with tp_comm_overlap, overlap the all-gather with the input-gradient matmul",
    ),
    Option(
        "tp_comm_bulk_dgrad",
        "model",
        bool,
        "with tp_comm_overlap, overlap the reduce-scatter with the"
        " weight-gradient matmul",
    ),
    Option(
        "tp_comm_overlap_ag",
        "model",
        bool,
        "with tp_comm_overlap, pipeline each all-gather with its matmul",
    ),
    Option(
        "tp_comm_overlap_rs",
        "model",
        bool,
        "with tp_comm_overlap, pipeline each reduce-scatter with its matmul",
    ),
    Option(
        "tp_comm_overlap_rs_dgrad",
        "model",
        bool,
        "with tp_comm_overlap, pipeline the reduce-scatter with the"
        " input-gradient matmul",
    ),
    # Recomputation.
    Option(
        "recompute_granularity",
        "model",
        str,
        "full: recompute whole layers in backward; selective: recompute"
        " only the modules recompute_modules names, in every layer",
        ("full", "selective"),
    ),
    Option(
        "recompute_method",
        "model",
        str,
        "with full recomputation, uniform: checkpoint every"
        " recompute_num_layers layers as one unit; block: recompute the"
        " first recompute_num_layers layers of each pipeline stage",
        ("uniform", "block"),
    ),
    Option(
        "recompute_num_layers",
        "model",
        int,
        "the layer count recompute_method takes (full recomputation); at least 1",
    ),
    Option(
        "recompute_modules",
        "model",
        str,
        "with selective recomputation, the modules recomputed, comma-separated;"
        " by default core_attn and layernorm, and moe_act in MoE models",
        RECOMPUTE_MODULES,
        form="names",
    ),
    Option(
        "distribute_saved_activations",
        "model",
        bool,
        "split the recomputed layers' saved inputs across the model-parallel"
        " ranks; not with sequence parallelism",
    ),
    # Activation offloading.
    Option(
        "cpu_offloading",
        "model",
        bool,
        "move the activations of the first cpu_offloading_num_layers layers"
        " to pinned host memory after their forward and back before their"
        " backward (Transformer Engine's layer-level offload); not with"
        " recomputation, pipeline parallelism or fine-grained offloading",
    ),
    Option(
        "cpu_offloading_num_layers",
        "model",
        int,
        "the layers cpu_offloading moves, at most one fewer than the model has",
    ),
    Option(
        "cpu_offloading_retain_pinned_cpu_buffers",
        "model",
        bool,
        "keep cpu_offloading's pinned host buffers across iterations instead"
        " of allocating them each step",
    ),
    Option(
        "fine_grained_activation_offloading",
        "model",
        bool,
        "move what the modules offload_modules names save for their backward"
        " to pinned host memory (module-level offload); not with"
        " cpu_offloading",
    ),
    Option(
        "offload_modules",
        "model",
        str,
        "with fine-grained offloading, the modules whose saved values move,"
        " comma-separated",
        OFFLOAD_MODULES,
        form="names",
    ),
    Option(
        "min_offloaded_tensor_size",
        "model",
        int,
        "with fine-grained offloading, the smallest tensor moved, in elements",
    ),
    Option(
        "activation_offload_fraction",
        "model",
        float,
        "with fine-grained offloading, the share of the eligible groups"
        " moved, from 0 to 1",
    ),
    Option(
        "delta_offload_bytes_across_pp_ranks",
        "model",
        int,
        "with fine-grained offloading, the difference in bytes moved between"
        " pipeline ranks, to balance them",
    ),
    Option(
        "fine_grained_offloading_max_inflight_offloads",
        "model",
        int,
        "with fine-grained offloading, the offloaded groups of one name whose"
        " copies may still be in flight before the compute stream waits for"
        " the oldest (0 waits after each; none never waits); a"
        " full-iteration CUDA graph needs it set",
    ),
    Option(
        "delay_offload_until_cuda_graph",
        "model",
        bool,
        "with fine-grained offloading under Transformer Engine's CUDA graphs,"
        " hold the experts' offloads (expert_fc1, moe_act) until the next"
        " graphed layer's replay returns instead of issuing them as the"
        " forward reaches them; the last graphed layer's are not offloaded",
    ),
    # The optimizer and its offloading.
    Option(
        "use_precision_aware_optimizer",
        "optimizer",
        bool,
        "step with Transformer Engine's fused Adam holding the masters and"
        " the moments at the contract's dtypes (Adam only; needs the"
        " distributed optimizer); the contract turns it on for masters or"
        " moments other than float32, and Megatron-LM's command line wants"
        " it beside optimizer_cpu_offload",
    ),
    Option(
        "optimizer_cpu_offload",
        "optimizer",
        bool,
        "keep optimizer_offload_fraction of the optimizer state, its"
        " float32 masters and its step on the host (Megatron-Core's"
        " HybridDeviceOptimizer, torch's AdamW there), the rest on the"
        " device",
    ),
    Option(
        "optimizer_offload_fraction",
        "optimizer",
        float,
        "the share of the optimizer state optimizer_cpu_offload keeps on"
        " the host, from 0 to 1 (Megatron's default, 0, keeps none)",
    ),
    Option(
        "overlap_cpu_optimizer_d2h_h2d",
        "optimizer",
        bool,
        "overlap the host optimizer's transfers with its step",
    ),
    Option(
        "pin_cpu_grads",
        "optimizer",
        bool,
        "pin the host copies of the gradients the host optimizer reads",
    ),
    Option(
        "pin_cpu_params",
        "optimizer",
        bool,
        "pin the float32 copies the host optimizer steps when it receives"
        " bfloat16 shards (the precision-aware path); a float32 shard's"
        " stepped copy is pinned either way",
    ),
    Option(
        "store_param_remainders",
        "optimizer",
        bool,
        "keep a float32 master as the bfloat16 weight and its low 16 bits"
        " (the precision-aware optimizer's masters)",
    ),
    Option(
        "optimizer_cuda_graph",
        "optimizer",
        bool,
        "capture the optimizer step in a CUDA graph (Transformer Engine's"
        " capturable Adam)",
    ),
    # Low precision.
    Option(
        "fp8",
        "mixed",
        str,
        "FP8 matmuls through Transformer Engine: e4m3 for every FP8"
        " tensor, or hybrid (e5m2 output gradients)",
        ("e4m3", "hybrid"),
    ),
    Option(
        "fp8_recipe",
        "mixed",
        str,
        "FP8 scaling: tensorwise (current scaling), delayed, mxfp8"
        " (Blackwell datacenter parts, not compute capability 12.0),"
        " blockwise, or custom (fp8_quantizer_factory's)",
        ("tensorwise", "delayed", "mxfp8", "blockwise", "custom"),
    ),
    Option(
        "fp8_quantizer_factory",
        "mixed",
        str,
        "with fp8_recipe custom, the Python import path of the callable that"
        " makes its quantizers (package.module.factory)",
    ),
    Option(
        "fp8_param_gather",
        "mixed",
        bool,
        "keep the compute weights in FP8 and gather them at FP8",
    ),
    Option(
        "fp8_wgrad",
        "mixed",
        bool,
        "compute the weight gradients at FP8 (off: at higher precision)",
    ),
    Option(
        "fp8_amax_history_len", "mixed", int, "delayed scaling's amax history length"
    ),
    Option(
        "fp8_amax_compute_algo",
        "mixed",
        str,
        "delayed scaling's amax: most_recent or max over the history",
        ("most_recent", "max"),
    ),
    Option("fp8_dot_product_attention", "mixed", bool, "FP8 dot-product attention"),
    Option("fp8_multi_head_attention", "mixed", bool, "FP8 multi-head attention"),
    Option(
        "first_last_layers_bf16",
        "mixed",
        bool,
        "keep the first and last layers at bfloat16 under FP8; not with"
        " delayed scaling",
    ),
    Option(
        "num_layers_at_start_in_bf16",
        "mixed",
        int,
        "with first_last_layers_bf16, the first layers kept at bfloat16",
    ),
    Option(
        "num_layers_at_end_in_bf16",
        "mixed",
        int,
        "with first_last_layers_bf16, the last layers kept at bfloat16",
    ),
    Option(
        "fp4",
        "mixed",
        str,
        "FP4 matmuls through Transformer Engine (e2m1; compute capability"
        " 10.0 and later); not with fp8",
        ("e2m1",),
    ),
    Option(
        "fp4_recipe",
        "mixed",
        str,
        "FP4 scaling: nvfp4 (NVFP4 block scaling, Bridge's default) or custom"
        " (fp4_quantizer_factory's)",
        ("nvfp4", "custom"),
    ),
    Option(
        "fp4_quantizer_factory",
        "mixed",
        str,
        "with fp4_recipe custom, the Python import path of the callable that"
        " makes its quantizers",
    ),
    Option(
        "fp4_param_gather",
        "mixed",
        bool,
        "keep the compute weights in FP4 and gather them at FP4",
    ),
    Option(
        "activation_func_fp8_input_store",
        "model",
        bool,
        "store the MLP activation's input at FP8 for backward (SwiGLU)",
    ),
    # MoE.
    Option(
        "moe_token_dispatcher_type",
        "model",
        str,
        "how tokens reach their experts: allgather, alltoall, or flex"
        " (DeepEP, HybridEP or NCCL EP; not on compute capability 12.0)",
        ("allgather", "alltoall", "flex"),
    ),
    Option(
        "moe_flex_dispatcher_backend",
        "model",
        str,
        "the flex dispatcher's backend: deepep (DeepEP's; Ampere, Hopper,"
        " B200/B300), deepepv2 (DeepEP's version-2 interface, which Bridge"
        " refuses under flex), hybridep (DeepEP's hybrid kernels; Ampere to"
        " Blackwell datacenter) or ncclep (Transformer Engine's NCCL EP;"
        " Hopper, Blackwell datacenter); the setup builds NCCL EP and DeepEP,"
        " deepep's and hybridep's, where 9.0 or 10.0 is a target",
        ("deepep", "deepepv2", "hybridep", "ncclep"),
    ),
    Option(
        "moe_grouped_gemm", "model", bool, "run a rank's experts as one grouped matmul"
    ),
    Option(
        "moe_use_grouped_tensor",
        "model",
        bool,
        "Transformer Engine's GroupedTensor path for the grouped matmul"
        " (capturable in CUDA graphs); needs moe_grouped_gemm",
    ),
    Option(
        "moe_permute_fusion",
        "model",
        bool,
        "fuse the token permutation of dispatch and combine",
    ),
    Option(
        "moe_router_fusion",
        "model",
        bool,
        "fuse the router's top-k and its balancing loss",
    ),
    Option(
        "moe_shared_expert_overlap",
        "model",
        bool,
        "overlap the shared experts with the dispatcher's communication"
        " (alltoall or flex, models with shared experts)",
    ),
    Option(
        "overlap_moe_expert_parallel_comm",
        "model",
        bool,
        "overlap expert-parallel all-to-alls with other microbatches'"
        " compute; needs expert parallelism, alltoall or flex, no full or"
        " MoE recomputation, and virtual stages under pipeline"
        " parallelism",
    ),
    Option(
        "delay_wgrad_compute",
        "model+ddp",
        bool,
        "delay the experts' weight-gradient matmuls to overlap more"
        " communication; needs overlap_moe_expert_parallel_comm (set on the"
        " model and on DDP, which hooks the delayed weights by its own copy)",
    ),
    Option(
        "moe_router_dtype",
        "model",
        str,
        "the router's and the expert outputs' weighted-sum dtype",
        ("fp32", "fp64"),
    ),
    # Attention and fusions.
    Option(
        "attention_backend",
        "model",
        str,
        "Transformer Engine's attention: auto (its choice), flash, fused,"
        " unfused, or local (Megatron-Core's PyTorch attention)",
        ("auto", "flash", "fused", "unfused", "local"),
    ),
    Option(
        "flash_attention_version",
        "model",
        int,
        "the FlashAttention generation Transformer Engine uses",
        (2, 3, 4),
    ),
    Option("apply_rope_fusion", "model", bool, "the fused rotary-embedding kernel"),
    Option(
        "bias_activation_fusion", "model", bool, "fuse the bias with the activation"
    ),
    Option(
        "bias_dropout_fusion", "model", bool, "fuse the bias, dropout and residual add"
    ),
    Option("masked_softmax_fusion", "model", bool, "the fused masked softmax"),
    Option(
        "persist_layer_norm",
        "model",
        bool,
        "the persistent fused layer-norm kernel (fixed hidden sizes)",
    ),
    Option(
        "fused_residual_rmsnorm",
        "model",
        bool,
        "fuse the residual add into RMSNorm's backward",
    ),
    Option(
        "cross_entropy_loss_fusion",
        "model",
        bool,
        "the fused vocabulary-parallel cross entropy over the logits",
    ),
    Option(
        "cross_entropy_fusion_impl",
        "model",
        str,
        "the fused cross entropy's implementation: native (Megatron-Core's)"
        " or te (Transformer Engine's parallel cross entropy)",
        ("native", "te"),
    ),
    Option(
        "gradient_accumulation_fusion",
        "model",
        bool,
        "add each weight gradient into the accumulated gradient as the"
        " matmul writes it (Transformer Engine's layers natively,"
        " Megatron-Core's own output layer through APEX's extension, which"
        " this environment builds); Bridge turns it on wherever Transformer"
        " Engine imports; for gradients narrower than float32 it is the"
        " contract's fuse_narrow_gradient_accumulation, which sets it",
    ),
    Option(
        "use_transformer_engine_op_fuser",
        "model",
        bool,
        "Transformer Engine's operation fuser in the submodules that take it",
    ),
    Option(
        "use_te_activation_func",
        "model",
        bool,
        "Transformer Engine's activation functions in the MLP",
    ),
    Option(
        "gdn_kernel_backend",
        "model",
        str,
        "Gated DeltaNet's kernels (Qwen 3.5): fla, transformer_engine or torch",
        ("fla", "transformer_engine", "torch"),
    ),
    Option(
        "disable_parameter_transpose_cache",
        "model",
        bool,
        "do not cache the weights' transposes between iterations",
    ),
    # CUDA graphs.
    Option(
        "cuda_graph_impl",
        "model",
        str,
        "CUDA graphs: none; local (Megatron-Core's, per layer or the whole"
        " iteration); transformer_engine (per layer); full_iteration"
        " (forward and backward as one graph, the optimizer outside)",
        ("none", "local", "transformer_engine", "full_iteration"),
    ),
    Option(
        "cuda_graph_modules",
        "model",
        str,
        "with per-layer graphs, the parts of a layer captured,"
        " comma-separated (none for the whole layer); mamba captures Mamba"
        " layers, of a family no catalog model is",
        ("attn", "mlp", "moe", "moe_router", "moe_preprocess", "mamba"),
        form="names",
    ),
    Option(
        "cuda_graph_warmup_steps",
        "model",
        int,
        "updates before the graphs are captured, fewer than the case's warmup",
    ),
    # The process environment.
    Option(
        "cuda_device_max_connections",
        "environment",
        int,
        "the hardware queues CUDA spreads a device's streams over"
        " (CUDA_DEVICE_MAX_CONNECTIONS, 1 to 32; unset is CUDA's 8): 1 keeps"
        " tensor-parallel communication ahead of the matmuls it overlaps, as"
        " Megatron advises with sequence parallelism; Megatron-FSDP refuses"
        " 1; CUDA graphs over many streams want 16 or more",
    ),
)

#: The options by name.
BY_NAME = {option.name: option for option in OPTIONS}

#: The updates Megatron runs before capturing CUDA graphs when a case names
#: no ``cuda_graph_warmup_steps``.
CUDA_GRAPH_WARMUP_STEPS = 3

#: The common settings Megatron's options leave alone.
LORA_OPTIONS = ("rank", "alpha", "factor_dtype", "head", "shared_experts")


def aware(precision) -> bool:
    """Whether the contract needs Megatron's precision-aware optimizer:
    masters or moments at another dtype than float32. Gradients accumulate
    at either dtype without it (``grad_reduce_in_fp32``), and with no master
    (Megatron-FSDP stepping the compute weights) Transformer Engine's fused
    Adam keeps its moments at float32, the only moments the contract
    accepts there."""

    if precision.master_dtype is None:
        return False
    moments = precision.resolved()["opt_state_dtype"]
    return precision.master_dtype != "float32" or moments not in (
        "float32",
        "default",
    )


def parse_value(option, value):
    """A setting's value as Megatron takes it: a list for a list form, a
    dtype for a dtype form, the value itself otherwise."""

    import torch

    if option.form == "names":
        return parse_list(value).split(",")
    if option.form == "integers":
        return [int(item) for item in parse_list(value).split(",")]
    if option.form == "dtype":
        return getattr(torch, value)
    return value


def parse_list(value):
    """A comma-separated list of names, kept as the string a case wrote."""
    names = [name.strip() for name in str(value).split(",") if name.strip()]
    if not names:
        raise ValueError("a list names at least one entry")
    return ",".join(names)


@dataclasses.dataclass(frozen=True)
class Base(common.Settings):
    CAPABILITIES: Capabilities = dataclasses.field(
        default=CAPABILITIES, init=False, repr=False, compare=False
    )

    implementation: str = IMPLEMENTATION
    # Megatron-Core recomputes what its configuration says; none leaves it
    # to Bridge or the recipe, a fraction recomputes that share of the
    # layers whole (full granularity, block method).
    recompute: float | None = None
    #: A Bridge recipe whose configuration the case starts from, by its
    #: name in ``megatron.bridge.recipes``; none starts from Bridge's
    #: defaults around the model of the Hugging Face configuration.
    recipe: str | None = None

    def request(self):
        """The catalog's model of the same architecture, by one of our
        implementations (PyTorch's where there is one): what the case is
        validated against and its useful work counted from."""

        from baselines.common.models import ModelRequest
        from workloads.models import catalog

        implementations = catalog.entry(self.model).implementations
        implementation = (
            "pytorch" if "pytorch" in implementations else implementations[0]
        )
        return ModelRequest(
            self.model, implementation, self.mode, self.lora, self.model_config
        )

    def describe(self, world_size):
        value = super().describe(world_size)
        value["model"]["implementation"] = self.implementation
        return value

    def model_parallel(self) -> int:
        return (
            (self.tensor_model_parallel_size or 1)
            * (self.pipeline_model_parallel_size or 1)
            * (self.context_parallel_size or 1)
        )

    def geometry(self, world_size) -> Geometry:
        """A replica is ``tensor x pipeline x context`` ranks."""

        if self.tokens_per_step is None:
            raise ValueError("a case names its tokens_per_step")
        split = self.model_parallel()
        if split < 1 or world_size % split:
            raise ValueError(
                f"{world_size} ranks do not form replicas of {split}"
                " (tensor x pipeline x context parallel)"
            )
        return Geometry(
            self.sequence_length,
            self.tokens_per_step,
            self.tokens_per_microbatch,
            world_size,
            None if split == 1 else world_size // split,
        )

    def resolve(self, world_size) -> dict[str, Any]:
        """The Hugging Face configuration NVIDIA's model is built from (the
        named checkpoint's own, or the catalog model's), and the modules
        selective recomputation recomputes."""

        entry = self.request().entry
        modules = {"recompute_modules": self.selective_modules(entry)}
        if self.checkpoint is not None:
            path = Path(self.checkpoint) / "config.json"
            return {**modules, "huggingface_config": json.loads(path.read_text())}
        family = import_module(f"workloads.models.{entry.family}.config")
        convert = getattr(family, "to_huggingface", None)
        if convert is None:
            raise Unsupported(
                f"workloads.models.{entry.family}.config has no to_huggingface,"
                " so Megatron cannot build its model"
            )
        return {
            **modules,
            "huggingface_config": convert(entry.config(**dict(self.model_config))),
        }

    def selective_modules(self, entry):
        """The modules selective recomputation recomputes: the case's, or by
        default core_attn and layernorm, with moe_act in a mixture of
        experts; none without selective recomputation."""

        if self.recompute_granularity != "selective":
            return None
        if self.recompute_modules is not None:
            return parse_list(self.recompute_modules).split(",")
        extra = MIXTURE_RECOMPUTE_MODULES if entry.mixture else ()
        return [*DEFAULT_RECOMPUTE_MODULES, *extra]

    def environment(self):
        """Transformer Engine picks its activation-offload implementation from
        ``NVTE_CPU_OFFLOAD_V1`` once, as it is imported, and Bridge refuses
        fine-grained offloading without the version-1 one, the one whose
        marks keep Transformer Engine's weights out of Megatron's offloads;
        CUDA reads ``CUDA_DEVICE_MAX_CONNECTIONS`` as it starts. The case
        sets each or leaves it unset, whatever the shell had."""

        connections = self.cuda_device_max_connections
        return {
            "NVTE_CPU_OFFLOAD_V1": (
                "1" if self.fine_grained_activation_offloading else None
            ),
            "CUDA_DEVICE_MAX_CONNECTIONS": (
                None if connections is None else str(connections)
            ),
        }

    def validate(self, world_size):
        # The options' values before the geometry reads the parallel sizes.
        self.check_values()
        return super().validate(world_size)

    def engine_validation(self, request, geometry, precision):
        """The engine's own rules: NVIDIA's implementation, option values of
        the types and choices Megatron takes, the shared settings and the
        precision contract as Megatron's options can say them, and CUDA
        graphs captured before the measured steps. What goes together in
        Megatron is Megatron's and Bridge's to decide as they set up: a case
        they refuse fails there with their message, and ``README.md``
        gathers their rules."""

        if self.implementation != IMPLEMENTATION:
            raise Unsupported(
                "Megatron trains NVIDIA's implementation; implementation must be"
                f" {IMPLEMENTATION}, not {self.implementation}"
            )
        if self.verify_reference:
            raise Unsupported(
                "the reference check compares our implementation's updates;"
                " Megatron trains NVIDIA's"
            )
        if self.recipe is not None and not self.recipe.isidentifier():
            raise ValueError(f"{self.recipe!r} is not a recipe name")
        if self.recompute is not None:
            # The share becomes Megatron's block method over the first layers
            # of each stage and chunk, counted from an even split; Megatron's
            # own recomputation options would say it again or otherwise, and
            # uneven stages give each a count of its own.
            named = [
                name
                for name in (
                    "recompute_granularity",
                    "recompute_method",
                    "recompute_num_layers",
                    "recompute_modules",
                    "num_layers_in_first_pipeline_stage",
                    "num_layers_in_last_pipeline_stage",
                    "account_for_embedding_in_pipeline_split",
                    "account_for_loss_in_pipeline_split",
                )
                if getattr(self, name) not in (None, False)
            ]
            if named:
                raise ValueError(
                    "recompute (a share of whole layers, from an even split of"
                    f" each stage) is set with {', '.join(named)}; set Megatron's"
                    " recomputation options alone, or the share alone"
                )
        if self.recompute_num_layers is not None and self.recompute_num_layers < 1:
            raise ValueError(
                "recompute_num_layers counts layers and is at least 1 (Megatron's"
                " uniform method never ends with fewer)"
            )
        self.check_contract(precision)
        if self.mode == "lora":
            unknown = set(self.lora) - set(LORA_OPTIONS)
            if unknown:
                raise Unsupported(
                    f"Megatron's LoRA takes {', '.join(LORA_OPTIONS)}; not"
                    f" {sorted(unknown)}"
                )
            if self.lora.get("head", "frozen") != "frozen":
                raise Unsupported("Megatron's LoRA leaves the head frozen")
            factor = self.lora.get("factor_dtype")
            if factor is not None and factor != precision.model_dtype:
                raise Unsupported(
                    "Megatron's LoRA keeps its factors at the model's dtype"
                    f" ({precision.model_dtype}): its adapters take the model's"
                    f" precision, not a factor_dtype of {factor}"
                )
        if self.cuda_graph_impl not in (None, "none") or self.optimizer_cuda_graph:
            # The graphs are captured in the update this many updates in
            # (Megatron's default is 3): the layers' and the whole
            # iteration's by Bridge and Megatron-Core, the optimizer step's
            # inside that update's step; the capture is no measured step.
            capture = self.cuda_graph_warmup_steps
            capture = CUDA_GRAPH_WARMUP_STEPS if capture is None else capture
            if capture >= self.warmup_steps:
                raise ValueError(
                    f"the CUDA graphs are captured at update {capture}, which must"
                    f" be one of the {self.warmup_steps} warmup updates; raise"
                    " warmup_steps or lower cuda_graph_warmup_steps"
                )

    def check_values(self):
        """Each option's value of its type, and among Megatron's choices."""

        for option in OPTIONS:
            value = getattr(self, option.name)
            if value is None:
                continue
            if option.kind is bool and not isinstance(value, bool):
                raise ValueError(f"{option.name} must be a boolean")
            if option.kind is int and type(value) is not int:
                raise ValueError(f"{option.name} must be an integer")
            if option.kind is float and type(value) not in (int, float):
                raise ValueError(f"{option.name} must be a number")
            if option.form == "names":
                names = parse_list(value).split(",")
                unknown = sorted(set(names) - set(option.choices))
                if unknown:
                    raise ValueError(
                        f"{option.name} takes {', '.join(option.choices)};"
                        f" not {unknown}"
                    )
            elif option.form == "integers":
                [int(item) for item in parse_list(value).split(",")]
            elif option.choices and value not in option.choices:
                raise ValueError(f"{option.name} must be one of {option.choices}")
        connections = self.cuda_device_max_connections
        if connections is not None and not 1 <= connections <= 32:
            raise ValueError("cuda_device_max_connections must be in [1, 32]")

    def fusing(self, precision):
        """Whether the weight-gradient matmuls add into the accumulated
        gradients as they write them: for gradients narrower than float32,
        the contract's fused narrow-gradient accumulation; for float32 ones,
        the case's gradient_accumulation_fusion, or none for Bridge's
        default (on wherever Transformer Engine imports)."""

        if precision.narrow_gradients:
            return precision.fuse_narrow_gradient_accumulation
        return self.gradient_accumulation_fusion

    def check_contract(self, precision):
        """The precision contract as Megatron's options can say it, and the
        case's own options where they would say otherwise."""

        sharded = self.use_megatron_fsdp
        gradients = precision.resolved()["grad_dtype"]
        if precision.master_dtype is None and not sharded:
            raise Unsupported(
                "Megatron-Core steps the weights themselves only under"
                " Megatron-FSDP (use_megatron_fsdp); its DDP optimizers keep masters"
            )
        if precision.master_dtype is None:
            moments = precision.resolved()["opt_state_dtype"]
            if moments not in ("default", "float32"):
                raise Unsupported(
                    "with no master, Megatron-FSDP's Adam (Transformer Engine's"
                    " fused Adam) keeps its moments at float32"
                )
        if not sharded and gradients not in (precision.model_dtype, "float32"):
            raise Unsupported(
                "Megatron's DDP accumulates gradients at the weights' dtype"
                f" ({precision.model_dtype}) or float32, not {gradients}"
            )
        if aware(precision) and self.use_precision_aware_optimizer is False:
            raise Unsupported(
                "this precision needs Megatron's precision-aware optimizer, which"
                " the case turns off"
            )
        if aware(precision) and self.use_distributed_optimizer is False and not sharded:
            raise Unsupported(
                "this precision needs Megatron's precision-aware optimizer, which"
                " needs the distributed optimizer the case turns off"
            )
        if (
            sharded
            and self.megatron_fsdp_version == 2
            and precision.narrow_gradients
            and precision.fuse_narrow_gradient_accumulation
        ):
            raise Unsupported(
                "Bridge turns gradient-accumulation fusion off under Megatron-FSDP"
                " version 2, so the contract's fused accumulation of narrow"
                " gradients is no run there; set fuse_narrow_gradient_accumulation"
                " false or use version 1"
            )
        fusion = self.gradient_accumulation_fusion
        if (
            precision.narrow_gradients
            and fusion is not None
            and fusion != precision.fuse_narrow_gradient_accumulation
        ):
            raise Unsupported(
                f"for {gradients} gradients gradient_accumulation_fusion is the"
                " contract's fuse_narrow_gradient_accumulation, and the two"
                " disagree; set the contract's"
            )


Settings = dataclasses.make_dataclass(
    "Settings",
    [
        (option.name, option.kind | None, dataclasses.field(default=None))
        for option in OPTIONS
    ],
    bases=(Base,),
    frozen=True,
    namespace={"__module__": __name__},
)
Settings.__doc__ = (
    "A Megatron-Core case: the common settings, the recipe it starts from,"
    " and Megatron's options."
)

HELP = {
    "recipe": "a Bridge recipe to start from, by its name in"
    " megatron.bridge.recipes; none builds from the Hugging Face configuration",
    "recompute": "share of each stage's layers recomputed whole (full granularity,"
    " block method); none leaves recomputation to Megatron's options",
    **{option.name: option.help for option in OPTIONS},
}


def add_arguments(parser):
    """The trainer's flags: the common ones, then Megatron's own."""

    common.add_arguments(
        parser,
        Settings,
        help=HELP,
        # A list's names are checked one by one when the case is validated.
        choices={
            option.name: option.choices
            for option in OPTIONS
            if option.choices and option.form == "value"
        },
        types={
            "recompute": float,
            **{
                option.name: parse_list
                if option.form in ("names", "integers")
                else str
                if option.form == "dtype"
                else option.kind
                for option in OPTIONS
                if option.kind is not bool
            },
        },
        booleans=frozenset(option.name for option in OPTIONS if option.kind is bool),
        implementations=(IMPLEMENTATION,),
    )


def from_namespace(arguments) -> Settings:
    """The settings a parsed trainer command line describes."""
    return common.from_namespace(Settings, arguments)


__all__ = [
    "BY_NAME",
    "CAPABILITIES",
    "ENVIRONMENT",
    "GEOMETRY_FIELDS",
    "IMPLEMENTATION",
    "MODEL_FIELDS",
    "OPTIONS",
    "PRECISION_FIELDS",
    "SOURCE_PATHS",
    "Option",
    "Settings",
    "add_arguments",
    "from_namespace",
]
