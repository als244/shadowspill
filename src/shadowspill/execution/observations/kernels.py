"""Allocation-free FP32 norm reductions into Executor-owned bounded buffers.

These eager diagnostic launches are not captured, profiled or stored as tasks.
The first kernel uses at most 256 partial sums, independent of tensor size.
"""

import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]

PARTIALS = 256
BLOCK = 4096


@triton.jit(do_not_specialize=["N"], do_not_specialize_on_alignment=["X"])
def norm_partials(
    X, Scratch, N, SHAPE: tl.constexpr, STRIDE: tl.constexpr, B: tl.constexpr
):
    total = tl.full((B,), 0, tl.float32)
    for start in range(tl.program_id(0) * B, N, tl.num_programs(0) * B):
        index = start + tl.arange(0, B)
        remaining = index
        offset = tl.full((B,), 0, tl.int64)
        for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
            offset += (remaining % SHAPE[axis]).to(tl.int64) * STRIDE[axis]
            remaining = remaining // SHAPE[axis]
        value = tl.load(X + offset, index < N, other=0).to(tl.float32)
        total += value * value
    tl.store(Scratch + tl.program_id(0), tl.sum(total, 0))


@triton.jit(do_not_specialize=["column", "count"])
def norm_sum(Scratch, Output, column, count, B: tl.constexpr):
    index = tl.arange(0, B)
    values = tl.load(Scratch + index, index < count, other=0)
    tl.store(Output + column, tl.sum(values, 0))


def grid(elements: int) -> int:
    return min(PARTIALS, max(1, triton.cdiv(elements, BLOCK)))
