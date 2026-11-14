"""automodel parallelism options, in command-line declaration order."""

from .types import CHECKPOINTING, DTYPES, SCOPES, STRATEGIES, _option

PARALLELISM = (
    _option(
        "strategy",
        "distributed",
        str,
        "the data-parallel implementation: fsdp2 (PyTorch's FSDP2),"
        " megatron_fsdp (Megatron-FSDP) or ddp (PyTorch's DDP)",
        STRATEGIES,
    ),
    _option("tp_size", "distributed", int, "tensor-parallel ranks"),
    _option("pp_size", "distributed", int, "pipeline stages"),
    _option("cp_size", "distributed", int, "context-parallel ranks"),
    _option(
        "ep_size",
        "distributed",
        int,
        "expert-parallel ranks, out of the non-pipeline ranks (MoE models)",
    ),
    _option(
        "dp_replicate_size",
        "distributed",
        int,
        "FSDP2's hybrid sharding: replicas of the sharded model (HSDP)",
    ),
    _option(
        "activation_checkpointing",
        "distributed",
        str,
        "none; full: every decoder layer recomputed; selective: every decoder"
        " layer recomputed under a policy that saves every other matmul,"
        " attention, top-k and the collectives",
        CHECKPOINTING,
    ),
    _option(
        "activation_checkpointing_scope",
        "distributed",
        str,
        "FSDP2 and DDP: the layer groups checkpointing wraps, comma-separated"
        " (all; or language, vision, audio of a multimodal model)",
        SCOPES,
        form="names",
    ),
    _option(
        "sequence_parallel",
        "distributed",
        bool,
        "the sequence-parallel variant of the tensor-parallel plan",
    ),
    _option(
        "tp_plan",
        "distributed",
        str,
        "a custom tensor-parallel plan: the import path of a plan or of a"
        " function returning one",
    ),
    _option(
        "defer_fsdp_grad_sync",
        "distributed",
        bool,
        "skip the reduce-scatter on all but the last microbatch, accumulating"
        " unsharded gradients",
    ),
    _option(
        "reshard_after_forward",
        "distributed",
        bool,
        "free each layer's gathered weights after its forward",
    ),
    _option(
        "enable_compile",
        "distributed",
        bool,
        "torch.compile each decoder layer after sharding",
    ),
    _option(
        "enable_fsdp2_prefetch",
        "distributed",
        bool,
        "explicit forward and backward prefetch chains between layers",
    ),
    _option(
        "fsdp2_backward_prefetch_depth",
        "distributed",
        int,
        "layers prefetched ahead in backward",
    ),
    _option(
        "fsdp2_forward_prefetch_depth",
        "distributed",
        int,
        "layers prefetched ahead in forward",
    ),
    _option(
        "enable_async_tensor_parallel",
        "distributed",
        bool,
        "asynchronous tensor parallelism (needs sequence parallelism)",
    ),
    _option(
        "patch_is_packed_sequence",
        "distributed",
        bool,
        "skip transformers' packed-sequence check, a host synchronization per"
        " attention layer",
    ),
    _option(
        "cpu_offload",
        "engine",
        bool,
        "FSDP2's CPUOffloadPolicy: weight shards, gradients and the optimizer"
        " step on the host (needs two or more ranks to shard)",
    ),
    _option(
        "offload_pin_memory",
        "engine",
        bool,
        "pin the offloaded shards' host memory",
    ),
    _option(
        "mp_param_dtype",
        "engine",
        str,
        "FSDP2's compute and all-gather dtype (mp_policy.param_dtype); the"
        " contract's model dtype",
        DTYPES,
        form="dtype",
    ),
    _option(
        "mp_reduce_dtype",
        "engine",
        str,
        "FSDP2's gradient reduce-scatter dtype (mp_policy.reduce_dtype); the"
        " contract's gradient dtype",
        DTYPES,
        form="dtype",
    ),
    _option(
        "mp_output_dtype",
        "engine",
        str,
        "FSDP2's cast of module outputs (mp_policy.output_dtype)",
        DTYPES,
        form="dtype",
    ),
    _option(
        "zero_dp_strategy",
        "distributed",
        int,
        "Megatron-FSDP's ZeRO level (3 shards the weights too)",
        (1, 2, 3),
    ),
    _option(
        "overlap_grad_reduce",
        "distributed",
        bool,
        "Megatron-FSDP: reduce gradients while backward continues",
    ),
    _option(
        "overlap_param_gather",
        "distributed",
        bool,
        "Megatron-FSDP: gather weights while forward continues",
    ),
    _option(
        "grad_reduce_in_fp32",
        "distributed",
        bool,
        "Megatron-FSDP: float32 main gradients; the contract's gradient dtype",
    ),
    _option(
        "preserve_fp32_weights",
        "distributed",
        bool,
        "Megatron-FSDP: float32 main weights; the contract's masters",
    ),
    _option(
        "check_for_nan_in_grad",
        "distributed",
        bool,
        "Megatron-FSDP: check the reduced gradients for NaN",
    ),
    _option(
        "average_in_collective",
        "distributed",
        bool,
        "Megatron-FSDP: average in the reduction itself",
    ),
    _option(
        "disable_bucketing",
        "distributed",
        bool,
        "Megatron-FSDP: one communication bucket per parameter",
    ),
    _option(
        "nccl_ub",
        "distributed",
        bool,
        "Megatron-FSDP: register its buffers as NCCL user buffers",
    ),
    _option(
        "fsdp_double_buffer",
        "distributed",
        bool,
        "Megatron-FSDP: keep its communication buffers, double buffered",
    ),
    _option(
        "init_fsdp_with_meta_device",
        "distributed",
        bool,
        "Megatron-FSDP: build the model on meta and materialize each rank's shards",
    ),
    _option(
        "bucket_cap_mb",
        "distributed",
        float,
        "DDP: megabytes per gradient bucket (PyTorch's default when none)",
    ),
    _option(
        "gradient_as_bucket_view",
        "distributed",
        bool,
        "DDP: gradients as views into the buckets",
    ),
    _option("static_graph", "distributed", bool, "DDP: a graph that does not change"),
    _option(
        "broadcast_buffers",
        "distributed",
        bool,
        "DDP: broadcast the buffers from the first rank every forward",
    ),
    _option(
        "find_unused_parameters",
        "distributed",
        bool,
        "DDP: find parameters a forward did not use",
    ),
    _option(
        "ignore_router_for_ac",
        "distributed.moe",
        bool,
        "under checkpointing, save the router's projection and top-k so the"
        " recomputation routes the same tokens",
    ),
    _option(
        "moe_reshard_after_forward",
        "distributed.moe",
        bool,
        "reshard MoE blocks, embeddings and the head after forward",
        key="reshard_after_forward",
    ),
    _option(
        "lm_head_precision",
        "distributed.moe",
        str,
        "the head's own FSDP unit at this dtype",
        DTYPES,
    ),
    _option(
        "wrap_outer_model",
        "distributed.moe",
        bool,
        "wrap the whole model in an FSDP unit of its own",
    ),
    _option(
        "pp_schedule",
        "distributed.pipeline",
        str,
        "PyTorch's pipeline schedule by name (1f1b, interleaved1f1b, gpipe, ...)",
    ),
    _option(
        "pp_schedule_csv",
        "distributed.pipeline",
        str,
        "a schedule from a CSV file in place of pp_schedule",
    ),
    _option(
        "pp_microbatch_size",
        "distributed.pipeline",
        int,
        "sequences per pipeline microbatch",
    ),
    _option(
        "layers_per_stage",
        "distributed.pipeline",
        int,
        "layers per (virtual) stage; none divides evenly",
    ),
    _option(
        "round_virtual_stages_to_pp_multiple",
        "distributed.pipeline",
        str,
        "round the virtual stages to a multiple of the stages",
        ("up", "down"),
    ),
    _option(
        "scale_grads_in_schedule",
        "distributed.pipeline",
        bool,
        "let the schedule scale the gradients",
    ),
    _option(
        "patch_inner_model",
        "distributed.pipeline",
        bool,
        "patch the inner model's forward for stages",
    ),
    _option(
        "patch_causal_lm_model",
        "distributed.pipeline",
        bool,
        "patch the causal-LM wrapper's forward for stages",
    ),
    _option(
        "pp_recv_buffer_pool",
        "distributed.pipeline",
        bool,
        "reuse the receive buffers between microbatches",
    ),
)
