"""automodel model options, in command-line declaration order."""

from .types import ATTENTION, DTYPES, LOSSES, _option

MODEL = (
    _option(
        "attn_implementation",
        "engine",
        str,
        "Hugging Face's attention: sdpa, eager, flash_attention_2, 3 or 4, or"
        " te (Transformer Engine's injected into Hugging Face's attention);"
        " AutoModel's default, flash_attention_2 where flash-attn imports and"
        " sdpa otherwise",
        ATTENTION,
    ),
    _option(
        "use_liger_kernel",
        "model",
        bool,
        "Liger's kernels in models built from Hugging Face's classes (not"
        " with tensor or context parallelism)",
    ),
    _option(
        "use_sdpa_patching",
        "model",
        bool,
        "run Hugging Face models' forward inside sdpa_kernel(sdpa_method)",
    ),
    _option(
        "force_hf",
        "model",
        bool,
        "Hugging Face's implementation even where AutoModel registers its own",
    ),
    _option(
        "sdpa_method",
        "engine",
        str,
        "the SDPA backends allowed, comma-separated",
        ("flash_attention", "efficient_attention", "math", "cudnn_attention"),
        form="names",
    ),
    _option(
        "attn",
        "model.backend",
        str,
        "the attention: torch, te, sdpa, flex, eager, tilelang, cudnn, fa4 or magi",
        ("torch", "te", "sdpa", "flex", "eager", "tilelang", "cudnn", "fa4", "magi"),
    ),
    _option(
        "sparse_attn",
        "model.backend",
        str,
        "sparse attention: generic, or msa (SM100 kernels of the models that"
        " provide them)",
        ("generic", "msa"),
    ),
    _option(
        "linear",
        "engine",
        str,
        "the linear layers: torch, te or quack; torch by default with the"
        " chunked loss, AutoModel's te or torch otherwise",
        ("torch", "te", "quack"),
    ),
    _option(
        "rms_norm",
        "model.backend",
        str,
        "the RMSNorm: torch, torch_fp32 (a compiled float32 norm), te or quack",
        ("torch", "torch_fp32", "te", "quack"),
    ),
    _option(
        "rope",
        "model.backend",
        str,
        "the rotary embedding: torch or quack",
        ("torch", "quack"),
    ),
    _option(
        "rope_fusion",
        "model.backend",
        bool,
        "Transformer Engine's fused rotary embedding",
    ),
    _option(
        "experts",
        "model.backend",
        str,
        "the experts: torch (a loop), torch_mm (torch._grouped_mm),"
        " torch_mm_mxfp8 or te (Transformer Engine's grouped linear)",
        ("torch", "te", "torch_mm", "torch_mm_mxfp8"),
    ),
    _option(
        "dispatcher",
        "model.backend",
        str,
        "the token dispatcher: torch, deepep, hybridep, uccl_ep or mok",
        ("torch", "deepep", "hybridep", "uccl_ep", "mok"),
    ),
    _option(
        "dispatcher_num_sms",
        "model.backend",
        int,
        "the streaming multiprocessors a DeepEP-family dispatcher takes",
    ),
    _option(
        "dispatcher_share_token_dispatcher",
        "model.backend",
        bool,
        "one communication manager shared by every MoE layer's flex dispatcher",
    ),
    _option(
        "dispatcher_async_dispatch",
        "model.backend",
        bool,
        "DeepEP and UCCL-EP dispatch and combine return asynchronously, their"
        " outputs on the communication stream",
    ),
    _option(
        "dispatcher_hybridep_permute_fusion",
        "model.backend",
        bool,
        "HybridEP: fuse token permutation into dispatch and combine",
    ),
    _option(
        "dispatcher_hybridep_compact_routing",
        "model.backend",
        bool,
        "HybridEP: send compact top-k indices instead of the routing map",
    ),
    _option(
        "dispatcher_hybridep_num_sms_preprocessing",
        "model.backend",
        int,
        "HybridEP: multiprocessors for routing-metadata preprocessing",
    ),
    _option(
        "dispatcher_hybridep_num_blocks_permute",
        "model.backend",
        int,
        "HybridEP: dispatch permutation blocks",
    ),
    _option(
        "dispatcher_hybridep_num_blocks_unpermute",
        "model.backend",
        int,
        "HybridEP: combine unpermutation blocks",
    ),
    _option(
        "mok_precision",
        "model.backend.mok",
        str,
        "Mixture-of-Kittens: its precision",
        ("bf16", "mxfp8"),
        key="precision",
    ),
    _option(
        "mok_fwd_num_comm_sms",
        "model.backend.mok",
        int,
        "Mixture-of-Kittens: forward communication multiprocessors",
        key="fwd_num_comm_sms",
    ),
    _option(
        "mok_bwd_num_comm_sms",
        "model.backend.mok",
        int,
        "Mixture-of-Kittens: backward communication multiprocessors",
        key="bwd_num_comm_sms",
    ),
    _option(
        "mok_minibatch_size",
        "model.backend.mok",
        int,
        "Mixture-of-Kittens: tokens per minibatch",
        key="minibatch_size",
    ),
    _option(
        "mok_macrobatch_size",
        "model.backend.mok",
        int,
        "Mixture-of-Kittens: tokens per macrobatch",
        key="macrobatch_size",
    ),
    _option(
        "mok_schedule_capacity_multiplier",
        "model.backend.mok",
        float,
        "Mixture-of-Kittens: schedule capacity multiplier",
        key="schedule_capacity_multiplier",
    ),
    _option(
        "mok_all_gather_top_experts_chunk_bytes",
        "model.backend.mok",
        int,
        "Mixture-of-Kittens: chunk bytes of the top-experts all-gather",
        key="all_gather_top_experts_chunk_bytes",
    ),
    _option(
        "fake_balanced_gate",
        "model.backend",
        bool,
        "replace the router with uniform assignment (a performance study,"
        " not training)",
    ),
    _option(
        "fake_gate_noise",
        "model.backend",
        float,
        "with the fake gate, 0 for perfectly balanced to 1 for random routing",
    ),
    _option(
        "enable_hf_state_dict_adapter",
        "model.backend",
        bool,
        "convert between Hugging Face's checkpoint layout and the model's",
    ),
    _option(
        "enable_fsdp_optimizations",
        "model.backend",
        bool,
        "MoE models: defer gradient sync and resharding to the last microbatch",
    ),
    _option(
        "te_fp8",
        "engine",
        str,
        "Transformer Engine's FP8 around every forward, by recipe: current,"
        " block or mxfp8 (linear te or experts te)",
        ("current", "block", "mxfp8"),
    ),
    _option(
        "gate_precision",
        "model.backend",
        str,
        "the router computation's dtype",
        DTYPES,
    ),
    _option(
        "compile_attn",
        "model.backend",
        bool,
        "compile the attention forward whole (attn sdpa, linear torch,"
        " rms_norm torch, no rope fusion)",
    ),
    _option(
        "compile_situ",
        "model.backend",
        bool,
        "compile the SiTU expert activation of the models that opt in",
    ),
    _option(
        "compile_norm",
        "model.backend",
        bool,
        "compile the float32 RMSNorm chain of the models that opt in",
    ),
    _option(
        "compile_hc",
        "model.backend",
        bool,
        "compile the hyper-connection cores of the models that opt in",
    ),
    _option(
        "shared_expert_overlap",
        "model.backend",
        bool,
        "run the shared experts of the models that opt in on a side stream",
    ),
    _option(
        "compile_router_weight",
        "model.backend",
        bool,
        "compile the float32 router-weight multiply of the grouped experts",
    ),
    _option(
        "benchmark_static_routing",
        "model.backend",
        bool,
        "with the fake gate at no noise, reuse the first microbatch's routing"
        " metadata (a performance study)",
    ),
    _option(
        "cuda_graph_modules",
        "engine",
        str,
        "the parts captured in CUDA graphs, comma-separated",
        ("attn", "te_dpa", "moe_router", "moe_preprocess"),
        form="names",
    ),
    _option(
        "loss",
        "engine",
        str,
        "the cross entropy: chunked (ChunkedCrossEntropy, the head and its"
        " loss chunk_len rows at a time, no full logits), masked"
        " (MaskedCrossEntropy over full logits), fused_linear"
        " (FusedLinearCrossEntropy) or te_parallel (TEParallelCrossEntropy over"
        " vocabulary-sharded logits); chunked by default, te_parallel under"
        " tensor parallelism",
        LOSSES,
    ),
    _option(
        "chunk_len",
        "engine",
        int,
        "chunked: token rows per chunk (512 by default)",
    ),
    _option(
        "chunked_compile",
        "engine",
        bool,
        "chunked: compile each chunk's projection and loss with their backward",
    ),
    _option(
        "fused_linear_impl",
        "engine",
        str,
        "fused_linear: cce (Cut Cross-Entropy, flat memory) or torch_compile"
        " (faster, memory growing with tokens times vocabulary)",
        ("cce", "torch_compile"),
    ),
    _option(
        "masked_fp32_upcast",
        "engine",
        bool,
        "masked: upcast the logits to float32",
    ),
)
