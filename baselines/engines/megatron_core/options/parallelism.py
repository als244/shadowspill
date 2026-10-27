"""megatron_core parallelism options, in command-line declaration order."""

from .types import DTYPES, SHARDING, Option

PARALLELISM = (
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
)
