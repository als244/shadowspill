"""The settings of one NeMo AutoModel case, shared by the launcher and the
trainer.

The harness trains NVIDIA's implementation of the catalog model's
architecture (AutoModel's own model where it registers one, Hugging Face's
otherwise) with AutoModel's fine-tuning recipe and its own training loop,
over the tokens every harness trains on. The model is built from the case's
Hugging Face configuration, the catalog model's from its family's
``to_huggingface`` or a named checkpoint's own. The shared settings (the
model, the geometry, the precision contract, the run) become AutoModel's
recipe sections; AutoModel's own options (``OPTIONS``) are settings under
AutoModel's names, each left to AutoModel's default unless the case sets it.
``README.md`` says what each does and which go together.
"""

from __future__ import annotations

import dataclasses
import json
from importlib import import_module
from pathlib import Path
from typing import Any

from baselines.common import settings as common
from baselines.common.precision import Capabilities, Unsupported

#: The sources whose digests identify this harness in a manifest and a row.
SOURCE_PATHS = (
    "baselines/__init__.py",
    "baselines/common",
    "baselines/automodel",
)

#: The environment beside the launcher's that the trainer runs in.
ENVIRONMENT = "nvidia-training"

#: The model block's name for NVIDIA's implementation.
IMPLEMENTATION = "automodel"

#: What AutoModel honors of the precision contract. Its weights are kept at
#: the storage dtype; float32 masters of bfloat16 weights come from FSDP2's
#: mixed precision over float32 storage when FSDP2 shards (two or more
#: ranks), or from Transformer Engine's fused Adam, which keeps float32 or
#: float16 masters and its moments at any of three dtypes on any number of
#: ranks. Gradients accumulate at the weights' dtype, at float32 only under
#: FSDP2's float32 storage. Autograd writes every weight gradient before it
#: is added, so a narrower one is added after.
CAPABILITIES = Capabilities(
    IMPLEMENTATION,
    model_dtypes=("bfloat16", "float32"),
    master_dtypes=(None, "float32", "float16"),
    grad_dtypes=(None, "float32", "bfloat16"),
    opt_state_dtypes=(None, "parameter", "float32", "float16", "bfloat16"),
    parameter_roundings=(None,),
    opt_state_roundings=(None,),
    fuse_narrow_gradient_accumulation=False,
)


@dataclasses.dataclass(frozen=True)
class Option:
    """One of AutoModel's options, as a case sets it: its name here, the
    recipe section that holds it (``distributed``; ``distributed.moe``;
    ``distributed.pipeline``; ``model``; ``model.backend``; ``loss_fn``;
    ``fp8``; ``compile``; ``step_scheduler``; ``peft``; ``optimizer``; or
    ``harness``, read by the harness to build a section), the key there when
    it differs from the name, the type of its value, what it does, its
    allowed values, and how a string is read (``names``, a comma-separated
    list of the allowed values; ``dtype``, a dtype's name)."""

    name: str
    section: str
    kind: type
    help: str
    choices: tuple = ()
    key: str | None = None
    form: str = "value"

    @property
    def field(self) -> str:
        return self.key or self.name


STRATEGIES = ("fsdp2", "megatron_fsdp", "ddp")
CHECKPOINTING = ("none", "full", "selective")
SCOPES = ("all", "language", "vision", "audio", "multimodal")
DTYPES = ("bfloat16", "float16", "float32")
LOSSES = ("chunked", "masked", "fused_linear", "te_parallel")
OPTIMIZERS = ("adamw", "adam", "fused_adam", "flash_adamw")
ATTENTION = (
    "sdpa",
    "eager",
    "flash_attention_2",
    "flash_attention_3",
    "flash_attention_4",
    "te",
)


def _option(name, section, kind, help, choices=(), key=None, form="value"):
    return Option(name, section, kind, help, tuple(choices), key, form)


#: Every AutoModel option a case can set, by subject; ``README.md``
#: describes each and what goes together.
OPTIONS = (
    # The strategy and the parallel sizes.
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
    # FSDP2.
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
        "harness",
        bool,
        "FSDP2's CPUOffloadPolicy: weight shards, gradients and the optimizer"
        " step on the host (needs two or more ranks to shard)",
    ),
    _option(
        "offload_pin_memory",
        "harness",
        bool,
        "pin the offloaded shards' host memory",
    ),
    _option(
        "mp_param_dtype",
        "harness",
        str,
        "FSDP2's compute and all-gather dtype (mp_policy.param_dtype); the"
        " contract's model dtype",
        DTYPES,
        form="dtype",
    ),
    _option(
        "mp_reduce_dtype",
        "harness",
        str,
        "FSDP2's gradient reduce-scatter dtype (mp_policy.reduce_dtype); the"
        " contract's gradient dtype",
        DTYPES,
        form="dtype",
    ),
    _option(
        "mp_output_dtype",
        "harness",
        str,
        "FSDP2's cast of module outputs (mp_policy.output_dtype)",
        DTYPES,
        form="dtype",
    ),
    # Megatron-FSDP.
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
    # DDP.
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
    # Mixture of experts under expert parallelism.
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
    # Pipeline parallelism.
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
    # The model's construction.
    _option(
        "attn_implementation",
        "harness",
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
        "harness",
        str,
        "the SDPA backends allowed, comma-separated",
        ("flash_attention", "efficient_attention", "math", "cudnn_attention"),
        form="names",
    ),
    # The backend of AutoModel's own models: every field of BackendConfig.
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
        "harness",
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
        "harness",
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
        "harness",
        str,
        "the parts captured in CUDA graphs, comma-separated",
        ("attn", "te_dpa", "moe_router", "moe_preprocess"),
        form="names",
    ),
    # The loss.
    _option(
        "loss",
        "harness",
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
        "harness",
        int,
        "chunked: token rows per chunk (512 by default)",
    ),
    _option(
        "chunked_compile",
        "harness",
        bool,
        "chunked: compile each chunk's projection and loss with their backward",
    ),
    _option(
        "fused_linear_impl",
        "harness",
        str,
        "fused_linear: cce (Cut Cross-Entropy, flat memory) or torch_compile"
        " (faster, memory growing with tokens times vocabulary)",
        ("cce", "torch_compile"),
    ),
    _option(
        "masked_fp32_upcast",
        "harness",
        bool,
        "masked: upcast the logits to float32",
    ),
    # torchao FP8.
    _option(
        "fp8",
        "harness",
        bool,
        "torchao's float8 training of the linear layers",
    ),
    _option(
        "fp8_recipe_name",
        "fp8",
        str,
        "torchao's scaling: tensorwise, rowwise or rowwise_with_gw_hp",
        ("tensorwise", "rowwise", "rowwise_with_gw_hp"),
        key="recipe_name",
    ),
    _option(
        "enable_fsdp_float8_all_gather",
        "fp8",
        bool,
        "gather FSDP2's weight shards at float8",
    ),
    _option(
        "precompute_float8_dynamic_scale_for_fsdp",
        "fp8",
        bool,
        "precompute the scales after each optimizer step",
    ),
    _option(
        "force_recompute_fp8_weight_in_bwd",
        "fp8",
        bool,
        "recompute the float8 weights in backward instead of keeping them",
    ),
    # Whole-model compilation (not under FSDP2).
    _option(
        "compile",
        "harness",
        bool,
        "torch.compile the whole model (not under FSDP2, which takes"
        " enable_compile, nor with pipeline parallelism)",
    ),
    _option(
        "compile_mode",
        "compile",
        str,
        "torch.compile's mode",
        ("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"),
        key="mode",
    ),
    _option("compile_fullgraph", "compile", bool, "compile one graph", key="fullgraph"),
    _option("compile_dynamic", "compile", bool, "dynamic shapes", key="dynamic"),
    _option(
        "compile_backend", "compile", str, "torch.compile's backend", key="backend"
    ),
    _option(
        "dynamo_cache_size_limit",
        "compile",
        int,
        "Dynamo's recompilation limit",
    ),
    # The step loop.
    _option(
        "gc_every_steps",
        "step_scheduler",
        int,
        "turn automatic garbage collection off and collect every this many updates",
    ),
    # The optimizer.
    _option(
        "optimizer",
        "harness",
        str,
        "AutoModel's optimizer: adamw (torch's AdamW), adam (torch's Adam),"
        " fused_adam (Transformer Engine's FusedAdam, with masters and moment"
        " dtypes) or flash_adamw (FlashAdamW, 24-bit masters); the contract"
        " chooses by default",
        OPTIMIZERS,
    ),
    _option(
        "adamw_fused",
        "harness",
        bool,
        "adamw and adam: torch's fused implementation (torch's default when none)",
    ),
    _option(
        "store_param_remainders",
        "harness",
        bool,
        "fused_adam: keep a float32 master as its bfloat16 weight and a 16-bit"
        " remainder",
    ),
    _option(
        "master_weight_bits",
        "harness",
        int,
        "flash_adamw: bits kept of each master",
    ),
    # LoRA.
    _option(
        "use_memory_efficient_lora",
        "peft",
        bool,
        "the custom autograd LoRA path and fused LoRA MLPs",
    ),
    _option("use_triton", "peft", bool, "Triton LoRA kernels"),
    _option(
        "moe_rank_scaling",
        "peft",
        bool,
        "expert adapters of rank dim // experts per token",
    ),
    _option("use_dora", "peft", bool, "DoRA on plain linear layers"),
    _option(
        "lora_dropout",
        "peft",
        float,
        "dropout on the adapters' path (the shared objective has none)",
        key="dropout",
    ),
    _option(
        "lora_dropout_position",
        "peft",
        str,
        "the dropout before or after the adapter",
        ("pre", "post"),
        key="dropout_position",
    ),
)

#: The options by name.
BY_NAME = {option.name: option for option in OPTIONS}

#: The common LoRA options AutoModel's adapter takes.
LORA_OPTIONS = ("rank", "alpha", "factor_dtype", "head", "shared_experts")


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
    # Autograd writes every weight gradient before AutoModel adds it, so a
    # narrower one is added after.
    fuse_narrow_gradient_accumulation: bool = False
    # AutoModel checkpoints every decoder layer or none: none leaves it to
    # activation_checkpointing, 0 is none and 1 is full.
    recompute: float | None = None

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
        return (self.tp_size or 1) * (self.pp_size or 1) * (self.cp_size or 1)

    def geometry(self, world_size):
        """A replica is ``tensor x pipeline x context`` ranks."""

        from baselines.common.geometry import Geometry

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
        """The Hugging Face configuration the model is built from (the named
        checkpoint's own, or the catalog model's), the loss and the linear
        layers' backend, and the precision mechanism the contract maps to on
        this many ranks."""

        resolved = {
            "loss": self.loss_kind(),
            "linear": self.linear_backend(),
            "precision_mechanism": self.mechanism(world_size),
            # Whether the model routes tokens to experts: a mixture AutoModel
            # builds from Hugging Face's classes trains no balancing loss.
            "mixture": self.request().entry.mixture,
        }
        if self.checkpoint is not None:
            path = Path(self.checkpoint) / "config.json"
            return {**resolved, "huggingface_config": json.loads(path.read_text())}
        entry = self.request().entry
        family = import_module(f"workloads.models.{entry.family}.config")
        convert = getattr(family, "to_huggingface", None)
        if convert is None:
            raise Unsupported(
                f"workloads.models.{entry.family}.config has no to_huggingface,"
                " so AutoModel cannot build its model"
            )
        return {
            **resolved,
            "huggingface_config": convert(entry.config(**dict(self.model_config))),
        }

    def loss_kind(self) -> str:
        """The case's loss, or the harness's: chunked, whose head and loss
        never hold the full logits, unless tensor parallelism shards the head
        over the vocabulary, where te_parallel is AutoModel's one loss that
        computes on the shards (the linear losses refuse a tensor-parallel
        head and masked gathers the whole logits)."""

        if self.loss is not None:
            return self.loss
        return "te_parallel" if (self.tp_size or 1) > 1 else "chunked"

    def linear_backend(self) -> str | None:
        """The case's linear layers, or torch's beside the chunked loss:
        AutoModel's models that build their head through the linear backend
        build Transformer Engine's where it imports, a head the chunked loss
        refuses, and AutoModel's own chunked recipes take torch's. None
        leaves AutoModel's default."""

        if self.linear is not None:
            return self.linear
        return "torch" if self.loss_kind() == "chunked" else None

    def mechanism(self, world_size) -> dict[str, Any]:
        """The AutoModel settings the contract maps to: the weights' storage
        dtype, the strategy's mixed precision (FSDP2's policy, Megatron-FSDP's
        float32 main weights and gradients), and the optimizer with its
        masters and moments. AutoModel parallelizes nothing on one rank:
        FSDP2 keeps the model as built, DDP and Megatron-FSDP cast it to
        bfloat16."""

        precision = self.precision()
        resolved = precision.resolved()
        model = precision.model_dtype
        gradients = resolved["grad_dtype"]
        moments = resolved["opt_state_dtype"]
        master = precision.master_dtype
        strategy = self.strategy or "fsdp2"
        # FSDP2 shards on any mesh of more than one rank; Megatron-FSDP only
        # where its data-parallel shard group spans more than one, and with
        # tensor, pipeline or context parallelism over every rank wraps
        # nothing.
        if strategy == "megatron_fsdp":
            sharded = world_size // self.model_parallel() > 1
        else:
            sharded = world_size > 1
        value = {
            "strategy": strategy,
            "storage": model,
            "optimizer": None,
            "master": None,
            "moments": None,
            "mp_param_dtype": model,
            "mp_reduce_dtype": gradients,
            "preserve_fp32_weights": None,
            "grad_reduce_in_fp32": None,
        }
        if model == "float32":
            # The weights are their own float32 masters.
            plain = moments in ("default", "float32")
            return {
                **value,
                "optimizer": self.optimizer or ("adamw" if plain else "fused_adam"),
                "moments": "float32" if plain else moments,
            }
        if strategy == "megatron_fsdp" and sharded:
            # Megatron-FSDP's own float32 main weights and gradients.
            return {
                **value,
                "optimizer": self.optimizer or "adamw",
                "moments": "float32" if master is not None else model,
                "preserve_fp32_weights": master == "float32",
                "grad_reduce_in_fp32": gradients == "float32",
                "master": master,
            }
        if master is None:
            plain = moments in ("default", model)
            return {
                **value,
                "optimizer": self.optimizer or ("adamw" if plain else "fused_adam"),
                "moments": model if plain else moments,
            }
        if gradients == "float32" and strategy == "fsdp2" and sharded:
            # AutoModel's float32 master weights: float32 storage computed at
            # the model dtype by FSDP2, gradients reduced and kept at float32.
            plain = moments in ("default", "float32")
            return {
                **value,
                "storage": "float32",
                "optimizer": self.optimizer or ("adamw" if plain else "fused_adam"),
                "moments": "float32" if plain else moments,
            }
        return {
            **value,
            "optimizer": self.optimizer or "fused_adam",
            "master": master,
            "moments": "float32" if moments == "default" else moments,
        }

    def validate(self, world_size):
        # The options' values before the geometry reads the parallel sizes,
        # and the checkpoint's kind before its weights are read.
        self.check_values()
        self.check_checkpoint()
        return super().validate(world_size)

    def check_checkpoint(self):
        """A checkpoint AutoModel would build more than the text model of:
        its configuration nests the text model's under ``text_config``."""

        if self.checkpoint is None:
            return
        config = json.loads((Path(self.checkpoint) / "config.json").read_text())
        if "text_config" in config:
            raise Unsupported(
                "the checkpoint is a model around a text model"
                f" ({', '.join(config.get('architectures') or [])}, the text"
                " model's configuration under text_config): AutoModel builds the"
                " whole of it, with what it adds (a vision tower, a"
                " multi-token-prediction layer whose loss joins the objective),"
                " and the shared objective is the text model's"
            )

    def harness_validation(self, request, geometry, precision):
        """The harness's own rules: NVIDIA's implementation, option values of
        the types and choices AutoModel takes, the shared settings and the
        precision contract as AutoModel's options can say them. What goes
        together in AutoModel is AutoModel's to decide as it sets up: a case
        it refuses fails there with its message, and ``README.md`` gathers
        its rules."""

        if self.implementation != IMPLEMENTATION:
            raise Unsupported(
                "AutoModel trains NVIDIA's implementation; implementation must be"
                f" {IMPLEMENTATION}, not {self.implementation}"
            )
        if self.verify_reference:
            raise Unsupported(
                "the reference check compares our implementation's updates;"
                " AutoModel trains NVIDIA's"
            )
        if self.recompute not in (None, 0, 1):
            raise Unsupported(
                "AutoModel checkpoints every decoder layer or none: recompute"
                " takes 0 or 1"
            )
        if self.recompute is not None and self.activation_checkpointing is not None:
            raise ValueError(
                "recompute and activation_checkpointing are two ways to say one"
                " thing; set one"
            )
        self.check_contract(precision, geometry.world_size)
        if self.mode == "lora":
            unknown = set(self.lora) - set(LORA_OPTIONS)
            if unknown:
                raise Unsupported(
                    f"AutoModel's LoRA takes {', '.join(LORA_OPTIONS)}; not"
                    f" {sorted(unknown)}"
                )
            if self.lora.get("head", "frozen") != "frozen":
                raise Unsupported("the harness's AutoModel LoRA leaves the head frozen")

    def check_values(self):
        """Each option's value of its type, and among AutoModel's choices."""

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
            elif option.choices and value not in option.choices:
                raise ValueError(f"{option.name} must be one of {option.choices}")

    def check_contract(self, precision, world_size):
        """The precision contract as AutoModel's options can say it, and the
        case's own options where they would say otherwise."""

        resolved = precision.resolved()
        gradients = resolved["grad_dtype"]
        mechanism = self.mechanism(world_size)
        strategy = mechanism["strategy"]
        if (
            strategy in ("ddp", "megatron_fsdp")
            and world_size == 1
            and precision.model_dtype != "bfloat16"
        ):
            raise Unsupported(
                f"AutoModel's {strategy} casts the model to bfloat16 on one"
                f" rank; a {precision.model_dtype} model takes fsdp2"
            )
        storage = mechanism["storage"]
        master = precision.master_dtype
        moments = resolved["opt_state_dtype"]
        # Each branch of the mapping holds the contract or is refused here:
        # nothing is substituted.
        if precision.model_dtype == "float32" and master not in (None, "float32"):
            raise Unsupported(
                "a float32 model's weights are their own float32 masters; masters"
                f" at {master} are no run of AutoModel's"
            )
        if precision.model_dtype == "float32" and gradients != "float32":
            raise Unsupported(
                "AutoModel keeps a float32 model's gradients at float32, not"
                f" {gradients}"
            )
        if (
            storage == "float32"
            and precision.model_dtype != "float32"
            and master
            not in (
                None,
                "float32",
            )
        ):
            raise Unsupported(
                "FSDP2's float32 storage is the float32 master of bfloat16 compute;"
                f" masters at {master} are not"
            )
        if mechanism["preserve_fp32_weights"] is not None:
            held = "float32" if master is not None else precision.model_dtype
            if moments not in ("default", held):
                raise Unsupported(
                    f"Megatron-FSDP's Adam keeps its moments at {held} here (float32"
                    " with masters, the weights' dtype without); not"
                    f" {moments}"
                )
        if strategy != "fsdp2":
            named = [
                name
                for name in (
                    "cpu_offload",
                    "offload_pin_memory",
                    "mp_param_dtype",
                    "mp_reduce_dtype",
                    "mp_output_dtype",
                )
                if getattr(self, name) is not None
            ]
            if named:
                raise Unsupported(
                    f"{', '.join(named)} are FSDP2's; AutoModel's {strategy} reads"
                    " none of them"
                )
        main_gradients = mechanism["grad_reduce_in_fp32"]
        if gradients == "float32" and storage != "float32" and not main_gradients:
            raise Unsupported(
                "AutoModel accumulates gradients at the weights' dtype; float32"
                " gradients of bfloat16 weights take FSDP2's float32 storage with"
                " float32 masters, or Megatron-FSDP's float32 main gradients,"
                " sharded across two or more data-parallel ranks"
            )
        optimizer = mechanism["optimizer"]
        own_masters = mechanism["preserve_fp32_weights"]
        if mechanism["master"] == "float16" and own_masters is not None:
            raise Unsupported(
                "Megatron-FSDP keeps float32 main weights or none; float16 masters"
                " take fsdp2 or ddp with fused_adam"
            )
        if (
            mechanism["master"] is not None
            and not own_masters
            and optimizer != "fused_adam"
        ):
            raise Unsupported(
                f"{optimizer} keeps no master of {storage} weights; the"
                " contract's masters take fused_adam"
            )
        factors = (self.lora or {}).get("factor_dtype")
        if (
            self.mode == "lora"
            and world_size == 1
            and factors not in (None, precision.model_dtype)
        ):
            raise Unsupported(
                "on one rank AutoModel casts each LoRA module, its adapters"
                f" included, to the compute dtype ({precision.model_dtype}); factors"
                f" at {factors} take two or more ranks"
            )
        if self.cpu_offload and optimizer == "fused_adam":
            raise Unsupported(
                "Transformer Engine's FusedAdam steps on the device, and FSDP2's"
                " CPU offload keeps the shards on the host; a contract that takes"
                " FusedAdam (masters, or moments at another dtype) takes no"
                " cpu_offload"
            )
        stepped = "float32" if own_masters else storage
        if optimizer in ("adamw", "adam") and mechanism["moments"] != stepped:
            raise Unsupported(
                f"torch's {optimizer} keeps its moments at the stepped weights'"
                f" dtype ({stepped}); other moments take fused_adam"
            )
        if optimizer == "flash_adamw":
            raise Unsupported(
                "FlashAdamW's 24-bit masters are not a dtype the contract names"
            )
        for name in (
            "mp_param_dtype",
            "mp_reduce_dtype",
            "preserve_fp32_weights",
            "grad_reduce_in_fp32",
        ):
            value = getattr(self, name)
            expected = mechanism[name]
            if value is not None and expected is not None and value != expected:
                raise Unsupported(
                    f"the contract sets {name} to {expected}; the case's"
                    f" {value} disagrees"
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
Settings.__doc__ = "An AutoModel case: the common settings and AutoModel's options."

HELP = {
    "recompute": "0 or 1: AutoModel's full activation checkpointing of every"
    " decoder layer, or none; none leaves it to activation_checkpointing",
    **{option.name: option.help for option in OPTIONS},
}


def add_arguments(parser):
    """The trainer's flags: the common ones, then AutoModel's own."""

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
                if option.form == "names"
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
    "IMPLEMENTATION",
    "OPTIONS",
    "SOURCE_PATHS",
    "Settings",
    "add_arguments",
    "from_namespace",
]
