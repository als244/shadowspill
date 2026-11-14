"""Meta-only model declaration; checkpoint payloads are read during execution."""

from pathlib import Path

import torch


def declare(
    config_type,
    checkpoint_type,
    model_type,
    *,
    checkpoint,
    tiny,
    initialization,
    gemm_precision,
    expert_storage,
    layer_limit,
    router_weight_dtype,
    lora,
):
    if initialization not in ("random", "checkpoint"):
        raise ValueError("initialization must be 'random' or 'checkpoint'")
    if tiny and initialization == "checkpoint":
        raise ValueError("tiny=True has no checkpoint weights to load")
    if (checkpoint is not None) == bool(tiny):
        raise ValueError("Choose a checkpoint directory or tiny=True")
    if router_weight_dtype not in (None, "bfloat16", "float32"):
        raise ValueError("router_weight_dtype must be None, bfloat16 or float32")
    if expert_storage not in (None, "bf16", "fp8", "fp8_block"):
        raise ValueError("expert_storage must be None, bf16, fp8 or fp8_block")
    router_dtype = (
        None if router_weight_dtype is None else getattr(torch, router_weight_dtype)
    )
    if tiny:
        if layer_limit is not None:
            raise ValueError("layer_limit applies to checkpoint prefixes")
        if gemm_precision not in ("bf16", "fp8"):
            raise ValueError("The tiny fixture's experts compute in bf16 or fp8")
        config = config_type.tiny(
            hidden_size=128,
            expert_storage=expert_storage
            or ("fp8_block" if gemm_precision == "fp8" else "bf16"),
            router_weight_dtype=router_dtype or torch.bfloat16,
            **lora,
        )
        return model_type(config, device="meta"), None
    source = checkpoint_type(
        Path(checkpoint).expanduser().resolve(),
        gemm_precision=gemm_precision,
        expert_storage=expert_storage,
    )
    try:
        model = source.build_model(
            layer_limit=layer_limit, router_weight_dtype=router_dtype, **lora
        )
        return model, source
    except BaseException:
        source.close()
        raise
