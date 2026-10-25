"""Checkpoint precision, copy coverage, and the resulting inventory."""

import torch
from torch import nn

from .experts import _projections, projection_keys
from .tensors import physical_state


def apply_dtypes(reader, model, router_weight_dtype):
    targets = physical_state(model)
    # The checkpoint, not a randomly initialized model's training defaults,
    # owns frozen storage precision. In particular, HF stores mHC projections
    # and KDA convolutions in BF16 even though their arithmetic may use FP32.
    # Resolve joined tensors as a unit so mixed source dtypes cannot silently
    # round one component. Quantized containers already declare their exact
    # physical component dtypes above.
    source_dtypes = {}
    for copy in reader.copies:
        dtype = reader.weights.source.tensor(copy.source, meta=True).dtype
        source_dtypes.setdefault(copy.destination, set()).add(dtype)
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    for name, dtypes in source_dtypes.items():
        if len(dtypes) != 1:
            raise ValueError(f"Mixed checkpoint dtypes for joined tensor {name}")
        dtype = next(iter(dtypes))
        if router_weight_dtype is not None and name.endswith(".mlp.router"):
            dtype = router_weight_dtype
        value = targets[name]
        if dtype == value.dtype:
            continue
        if name not in parameters and name not in buffers:
            raise ValueError(f"Checkpoint dtype differs for physical component {name}")
        if not value.is_floating_point() or not dtype.is_floating_point:
            raise ValueError(
                f"Unsupported checkpoint storage dtype for {name}: {dtype}"
            )
        parent, _, leaf = name.rpartition(".")
        owner = model.get_submodule(parent)
        value = torch.empty_like(value, dtype=dtype)
        if name in parameters:
            value = nn.Parameter(value, requires_grad=parameters[name].requires_grad)
        setattr(owner, leaf, value)
    targets = physical_state(model)
    return targets


def validate_copies(reader, targets, router_weight_dtype):
    for copy in reader.copies:
        source = reader.weights.source.tensor(copy.source, meta=True)
        target = copy.view(targets)
        if source.shape != target.shape:
            raise ValueError(
                f"{copy.source}: {tuple(source.shape)} does not match "
                f"{copy.destination}: {tuple(target.shape)}"
            )
        router_override = (
            router_weight_dtype is not None
            and copy.destination.endswith(".mlp.router")
            and source.dtype in (torch.bfloat16, torch.float32)
            and target.dtype == router_weight_dtype
        )
        if source.dtype != target.dtype and not router_override:
            raise ValueError(
                f"Unsupported dtype conversion {copy.source}: "
                f"{source.dtype} -> {target.dtype}"
            )


def used_keys(reader, c, include_vision):
    used = {copy.source for copy in reader.copies}
    # Every tensor of an expert projection is accounted for, the
    # activation scales of an NVFP4 recipe included: the FP8 forms
    # quantize their activations per row per call.
    used.update(
        key
        for conversion in reader.conversions
        for _index, source in conversion.experts
        for projection in _projections(conversion.kind)
        for key in projection_keys(reader, f"{source}.{projection}")
    )
    used.update(reader._expert_keys)
    required = {
        key
        for key in reader.weights.files
        if key.startswith(
            (
                reader.source_prefix + "embed_tokens.",
                reader.source_prefix + "norm.",
                "lm_head.",
            )
        )
    }
    for i in range(c.num_hidden_layers):
        required.update(
            key
            for key in reader.weights.files
            if key.startswith(f"{reader.source_prefix}layers.{i}.")
        )
    if include_vision:
        required.update(
            key for key in reader.weights.files if key.startswith("model.visual.")
        )
    if required - used:
        raise ValueError(f"Unmapped model tensors: {sorted(required - used)[:10]}")
    return used


def inventory(reader, model, c, vision, targets, used):
    return {
        "layers": c.num_hidden_layers,
        "vision_layers": 0 if vision is None else vision.depth,
        "source_tensors": len(used),
        "physical_tensors": len(targets),
        "model_bytes": sum(t.numel() * t.element_size() for t in targets.values()),
        "largest_tensor_bytes": max(
            t.numel() * t.element_size() for t in targets.values()
        ),
        "expert_storage": c.expert_storage,
        "checkpoint_expert_format": reader.expert_format,
        "converted_expert_banks": len(reader.conversions),
        "storage_dtype_bytes": {
            str(dtype): sum(
                t.numel() * t.element_size()
                for t in targets.values()
                if t.dtype == dtype
            )
            for dtype in sorted({t.dtype for t in targets.values()}, key=str)
        },
        "checkpoint_storage_preserved": not reader.conversions,
        "gemm_precision": c.gemm_precision,
        "lora_rank": c.lora_rank,
        "lora_alpha": c.lora_alpha,
        "lora_factor_dtype": str(c.lora_factor_dtype),
        "lora_head": c.lora_head,
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "ignored_source_tensors": len(reader.weights.files.keys() - used),
    }
