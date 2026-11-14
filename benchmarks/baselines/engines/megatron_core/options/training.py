"""megatron_core training options, in command-line declaration order."""

from .types import OFFLOAD_MODULES, RECOMPUTE_MODULES, Option

TRAINING = (
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
)
