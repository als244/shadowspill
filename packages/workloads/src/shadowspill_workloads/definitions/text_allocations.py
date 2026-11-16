"""Exact opaque-call allocation contracts for the supplied packed Llama path.

These declarations cover fixed geometry/layout and captured scalar arguments,
for every tensor value admitted by the schema. They do not supply byte counts:
the ordinary compiler/profiler and joint admission still own those. The model
source pin includes this file and the MLOps implementation closure; the Program
also pins the Torch toolchain. Unknown implementations remain rejected.

Embedding sorts a fixed number of indices and allocates a full declared table
gradient, regardless of token frequency. RMSNorm allocates one output/statistic
per row and ceil(rows/64)*width FP32 backward partials. SwiGLU and table RoPE use
input-sized outputs and fixed-layout copies. Searchsorted's output and optional
contiguous copies depend only on tensor extents/layout, not boundary values.

Native variable-length attention sizes output/LSE and backward scratch from
total rows, heads, head width, maximum length and the *allocated* sequence slots.
In particular, native FA2 backward uses total_q + 128*slots rows, even when some
slots are empty. FA3 uses shape-bounded scheduler metadata, padded row totals,
splits and semaphores; device-computed scheduling never resizes those buffers.
Hardware/provider qualification is still required for each deployed path.

The FP32 packed attention path can select ATen efficient-attention backward.
Its CUDA implementation allocates Q/K/V gradients, an aligned optional bias
gradient, row deltas and split-key workspace from tensor geometry, captured
flags and device properties. Mask contents and RNG values do not resize them.
Large-batch chunking likewise uses the batch extent. The Torch toolchain pin
identifies this implementation; other attention entry points need their own
contracts.
"""

CONTRACTS = {
    **{
        f"torch.ops.mlops.{name}_{direction}.default": (
            "shadowspill/llama-fixed-provider-allocation@1"
        )
        for name in (
            "embedding_builtin_deterministic",
            "rms_norm_builtin_triton",
            "swiglu_builtin_triton",
            "rope_builtin_triton_table",
            "flash_attention_builtin_aten",
        )
        for direction in ("fwd", "bwd")
    },
    "torch.ops.aten.searchsorted.Tensor": "shadowspill/aten-searchsorted-allocation@1",
    "torch.ops.aten._scaled_dot_product_efficient_attention_backward.default": (
        "shadowspill/aten-efficient-attention-backward-allocation@1"
    ),
}
