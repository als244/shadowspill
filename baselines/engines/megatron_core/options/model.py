"""megatron_core model options, in command-line declaration order."""

from .types import Option

MODEL = (
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
