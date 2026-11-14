"""automodel training options, in command-line declaration order."""

from .types import OPTIMIZERS, _option

TRAINING = (
    _option(
        "fp8",
        "engine",
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
    _option(
        "compile",
        "engine",
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
    _option(
        "gc_every_steps",
        "step_scheduler",
        int,
        "turn automatic garbage collection off and collect every this many updates",
    ),
    _option(
        "optimizer",
        "engine",
        str,
        "AutoModel's optimizer: adamw (torch's AdamW), adam (torch's Adam),"
        " fused_adam (Transformer Engine's FusedAdam, with masters and moment"
        " dtypes) or flash_adamw (FlashAdamW, 24-bit masters); the contract"
        " chooses by default",
        OPTIMIZERS,
    ),
    _option(
        "adamw_fused",
        "engine",
        bool,
        "adamw and adam: torch's fused implementation (torch's default when none)",
    ),
    _option(
        "store_param_remainders",
        "engine",
        bool,
        "fused_adam: keep a float32 master as its bfloat16 weight and a 16-bit"
        " remainder",
    ),
    _option(
        "master_weight_bits",
        "engine",
        int,
        "flash_adamw: bits kept of each master",
    ),
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
