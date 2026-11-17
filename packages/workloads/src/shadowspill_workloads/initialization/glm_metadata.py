"""Compact, framework-free GLM checkpoint storage declarations."""

DTYPES = {
    "BF16": "bfloat16",
    "F16": "float16",
    "F32": "float32",
    "F8_E4M3": "float8_e4m3fn",
}


def describe(metadata, tensors, architecture, *, prefix):
    """Keep ordinary tensor precision and native expert storage in saved settings.

    Only dtype names are retained for dense projections/parameters. Expert tensor
    declarations are checked here, then discarded: the architecture already
    describes their stacked geometry. No checkpoint path or payload enters builds.
    """
    quant = metadata.get("quantization_config") or metadata.get("text_config", {}).get(
        "quantization_config", {}
    )
    dtypes, expert_dtypes = {}, set()
    for name, tensor in tensors.items():
        if not name.startswith((prefix, "lm_head.")):
            continue
        if ".mlp.experts." in name:
            if name.endswith(".weight"):
                expert_dtypes.add(tensor["dtype"])
            elif not name.endswith(".weight_scale_inv"):
                raise ValueError(
                    "public GLM initialization requires dense or native "
                    "block-FP8 experts"
                )
            continue
        if tensor["dtype"] not in DTYPES:
            raise ValueError(
                f"unsupported GLM checkpoint storage: {name}: {tensor['dtype']}"
            )
        dtypes[name] = DTYPES[tensor["dtype"]]
    if expert_dtypes not in ({"BF16"}, {"F8_E4M3"}):
        raise ValueError("GLM expert banks require uniform BF16 or native E4M3 storage")
    if ("F8_E4M3" in expert_dtypes or "float8_e4m3fn" in dtypes.values()) and (
        quant.get("quant_method") != "fp8"
        or quant.get("fmt", "e4m3") != "e4m3"
        or quant.get("weight_block_size") != [128, 128]
    ):
        raise ValueError(
            "GLM FP8 storage requires explicit E4M3 128x128 block encoding"
        )
    return {
        "architecture": {
            **architecture,
            "expert_storage": "fp8_block" if "F8_E4M3" in expert_dtypes else "bf16",
        },
        "storage": {"tensor_dtypes": dtypes},
    }
